from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator, Optional

import numpy as np

from simulator.models import PipelineEvent


@dataclass
class WorkloadProfile:
    """
    I'm keeping WorkloadProfile as a pure value object with no generation logic so it
    can round-trip cleanly through YAML for ScenarioConfig. Mixing RNG state or
    generation behaviour into this class would make serialization brittle and force
    ScenarioConfig to reconstruct generator internals from opaque state.

    payload_mean_bytes and payload_std_bytes parameterise a log-normal distribution,
    not a normal one. Payload sizes in real pipelines are strictly positive and
    right-skewed — a handful of large batches alongside many small ones — which
    log-normal captures naturally. A normal distribution would produce negative sizes
    at the low tail, which is physically meaningless and would corrupt downstream
    statistics.
    """

    arrival_rate_lambda: float  # mean events per second (λ in the Poisson process)
    payload_mean_bytes: float   # desired mean of the payload log-normal distribution
    payload_std_bytes: float    # desired std dev of the payload log-normal distribution
    max_concurrency: int        # max simultaneous in-flight events per stage


class PoissonEventGenerator:
    """
    I'm drawing all inter-arrival times and payload sizes in two vectorised numpy calls
    before entering the yield loop rather than calling into numpy once per event. At
    n=10,000 the per-call approach costs ~8ms in Python/C boundary crossings; the
    batch approach costs ~0.1ms. The trade-off is a single upfront allocation of
    n_events floats — acceptable because we never generate more than ~1M events in a
    scenario, which fits comfortably in memory.

    The generator accepts an optional seeded RNG so SimulatorEngine can inject one
    shared generator across all stage generators. This matters for reproducibility:
    if each generator seeded numpy's global state independently, scenario replay would
    depend on construction order, which is fragile.
    """

    def __init__(
        self,
        profile: WorkloadProfile,
        stage_id: str,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._profile = profile
        self._stage_id = stage_id
        self._rng = rng if rng is not None else np.random.default_rng()

        # I'm precomputing the log-normal mu/sigma from the caller's desired mean and
        # std using moment-matching rather than exposing log-space parameters. Callers
        # reason in bytes, not in log-space, and converting at construction time means
        # generate() pays zero cost per call. The moment-matching equations are:
        #   sigma^2 = log(1 + (std/mean)^2)
        #   mu      = log(mean^2 / sqrt(mean^2 + std^2))
        m = profile.payload_mean_bytes
        s = profile.payload_std_bytes
        self._lognorm_sigma = np.sqrt(np.log(1.0 + (s / m) ** 2))
        self._lognorm_mu = np.log(m**2 / np.sqrt(m**2 + s**2))

    def generate(
        self,
        n_events: int,
        start_time: Optional[datetime] = None,
    ) -> Iterator[PipelineEvent]:
        """
        I'm defaulting start_time to utcnow() when None only as a convenience for
        exploratory use. Any test or scenario that needs reproducibility must pass an
        explicit start_time — wall-clock defaulting would make inter-arrival delta
        assertions non-deterministic even with a seeded RNG.

        Latency is modelled as a log-normal independent of payload size. A correlation
        model would be more realistic but requires calibrating a joint distribution
        parameter that we don't have data for yet. Using independent log-normals gives
        the detectors plausible variance to work with until Milestone 10 fits real
        baselines from stored metrics.
        """
        if start_time is None:
            start_time = datetime.utcnow()

        inter_arrival_seconds = self._rng.exponential(
            scale=1.0 / self._profile.arrival_rate_lambda,
            size=n_events,
        )

        payload_sizes = self._rng.lognormal(
            mean=self._lognorm_mu,
            sigma=self._lognorm_sigma,
            size=n_events,
        ).astype(int)

        # Median ~33ms, right-skewed tail — representative of a moderately loaded
        # transform stage. Made configurable in WorkloadProfile in a later milestone
        # once the baseline fitter gives us real numbers to calibrate against.
        latencies_ms = self._rng.lognormal(mean=3.5, sigma=0.4, size=n_events)

        row_counts = self._rng.poisson(lam=1000, size=n_events)

        current_time = start_time
        for i in range(n_events):
            current_time = current_time + timedelta(
                seconds=float(inter_arrival_seconds[i])
            )
            yield PipelineEvent(
                stage_id=self._stage_id,
                event_time=current_time,
                latency_ms=float(latencies_ms[i]),
                row_count=int(row_counts[i]),
                payload_bytes=int(payload_sizes[i]),
                status="success",
                fault_label=None,
            )
