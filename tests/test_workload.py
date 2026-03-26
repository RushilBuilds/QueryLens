from datetime import datetime

import numpy as np

from simulator.workload import PoissonEventGenerator, WorkloadProfile


def test_inter_arrival_rate_converges_to_lambda() -> None:
    """
    n=10,000: relative error of an exponential sample mean scales as 1/sqrt(n),
    giving ~1% relative std at this count. A 5% tolerance is roughly 5 sigma of
    headroom. Fewer samples risk flakiness; more slow the suite with no meaningful
    tightening.
    """
    arrival_rate = 5.0  # events per second
    profile = WorkloadProfile(
        arrival_rate_lambda=arrival_rate,
        payload_mean_bytes=1024.0,
        payload_std_bytes=256.0,
        max_concurrency=4,
    )
    rng = np.random.default_rng(seed=42)
    generator = PoissonEventGenerator(profile=profile, stage_id="stage_source", rng=rng)

    events = list(
        generator.generate(n_events=10_000, start_time=datetime(2024, 1, 1, 0, 0, 0))
    )

    # Reconstruct inter-arrival times from consecutive event_time deltas rather than
    # reading internal generator state — this tests the observable contract, not the
    # implementation detail.
    timestamps = [e.event_time for e in events]
    inter_arrival_seconds = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]

    expected_mean = 1.0 / arrival_rate
    observed_mean = float(np.mean(inter_arrival_seconds))

    relative_error = abs(observed_mean - expected_mean) / expected_mean
    assert relative_error < 0.05, (
        f"Inter-arrival mean {observed_mean:.5f}s deviates {relative_error:.1%} from "
        f"expected {expected_mean:.5f}s — Poisson process is not converging to λ={arrival_rate}"
    )


def test_payload_sizes_follow_lognormal_distribution() -> None:
    """
    First and second moments checked rather than a KS test: the KS test requires the
    exact log-normal mu/sigma, which means inverting the generator's moment-matching
    equations — testing the math against itself, not the implementation. Observed mean
    and std within 10% of configured values catches the parameter conversion bugs that
    occur in practice (swapped mu/sigma, incorrect log-space variance).

    Integer truncation of payload_bytes introduces ~0.01% negative bias at mean=4096 —
    well within the 10% tolerance.
    """
    payload_mean = 4096.0  # bytes
    payload_std = 1024.0   # bytes
    profile = WorkloadProfile(
        arrival_rate_lambda=10.0,
        payload_mean_bytes=payload_mean,
        payload_std_bytes=payload_std,
        max_concurrency=4,
    )
    rng = np.random.default_rng(seed=99)
    generator = PoissonEventGenerator(profile=profile, stage_id="stage_transform", rng=rng)

    events = list(
        generator.generate(n_events=10_000, start_time=datetime(2024, 1, 1, 0, 0, 0))
    )

    payload_sizes = np.array([e.payload_bytes for e in events], dtype=float)
    observed_mean = float(np.mean(payload_sizes))
    observed_std = float(np.std(payload_sizes))

    mean_error = abs(observed_mean - payload_mean) / payload_mean
    std_error = abs(observed_std - payload_std) / payload_std

    assert mean_error < 0.10, (
        f"Payload mean {observed_mean:.1f} bytes deviates {mean_error:.1%} from "
        f"configured {payload_mean:.1f} bytes"
    )
    assert std_error < 0.10, (
        f"Payload std {observed_std:.1f} bytes deviates {std_error:.1%} from "
        f"configured {payload_std:.1f} bytes"
    )
