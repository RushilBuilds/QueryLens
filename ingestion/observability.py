from __future__ import annotations

import logging
import threading
from wsgiref.simple_server import WSGIRequestHandler, make_server

import prometheus_client as prom
import structlog
from prometheus_client import make_wsgi_app

# ---------------------------------------------------------------------------
# Prometheus metrics
#
# Defined at module level as singletons: prometheus_client raises ValueError
# if two collectors share the same name in the same registry, which would
# happen on every MetricConsumer or IngestionWorker construction in tests.
# Module-level singletons register exactly once at import time.
# ---------------------------------------------------------------------------

RECORDS_CONSUMED = prom.Counter(
    "ingestion_records_consumed_total",
    "Total records successfully parsed from the Redpanda topic, labelled by stage.",
    ["stage_id"],
)

RECORDS_WRITTEN = prom.Counter(
    "ingestion_records_written_total",
    "Total records committed to PostgreSQL, labelled by stage.",
    ["stage_id"],
)

# DLQ events are not labelled by stage_id: a message lands in the DLQ because
# it couldn't be deserialised, so no stage_id is available. stage_id="unknown"
# would look meaningful but carry no actual information.
DLQ_EVENTS = prom.Counter(
    "ingestion_dlq_events_total",
    "Total records routed to the DLQ due to deserialisation failure.",
)

WRITE_LATENCY = prom.Histogram(
    "ingestion_write_latency_seconds",
    "Wall time for a single PostgreSQL batch INSERT, in seconds.",
    # Sub-10ms buckets at the low end: target p99 for a 500-record batch is
    # under 50ms. Default prometheus_client buckets top out at 10s and would
    # compress all normal traffic into the first two buckets.
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

CONSUMER_LAG = prom.Gauge(
    "ingestion_consumer_lag_seconds",
    "Age of the oldest event in the current batch (wall_time - event_time), per stage.",
    ["stage_id"],
)


# ---------------------------------------------------------------------------
# structlog configuration
# ---------------------------------------------------------------------------


def configure_structlog() -> None:
    """
    Configures structlog with JSON output rather than key=value. The ingestion
    layer runs in a container whose stdout is scraped by a log aggregator
    (Loki, CloudWatch, etc.); JSON lets the aggregator parse fields without a
    custom regex, making trace_id and stage_id filterable dimensions automatically.

    PrintLoggerFactory (stdout) instead of stdlib logging: no existing logging
    hierarchy to integrate with, and the stdlib bridge adds handler-dispatch
    latency for no benefit.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# MetricsServer
# ---------------------------------------------------------------------------


class _SilentHandler(WSGIRequestHandler):
    """
    Suppresses the per-request access log written to stderr by default.
    Prometheus scrapes every 15 seconds — 4 log lines per minute of noise that
    drowns real events in production and clutters the pytest capture buffer in tests.
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class MetricsServer:
    """
    Uses wsgiref.simple_server rather than prometheus_client.start_http_server:
    start_http_server's return type changed between 0.16 and 0.20 (None → tuple),
    making shutdown() unportable. wsgiref is stdlib and gives clean lifecycle
    control: start() serves in a daemon thread, stop() calls httpd.shutdown()
    synchronously so tests tear down without a sleep.

    Bound to 127.0.0.1 only — /metrics should be reachable by a local Prometheus
    sidecar, not exposed to the container network.
    """

    def __init__(self, port: int = 0) -> None:
        # port=0: OS assigns a free port, preventing collisions in parallel tests.
        self._httpd = make_server("127.0.0.1", port, make_wsgi_app(), handler_class=_SilentHandler)
        self._port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="metrics-server",
        )

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
