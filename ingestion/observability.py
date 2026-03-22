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
# I'm defining all metrics at module level as singletons rather than
# instantiating them inside classes. prometheus_client raises a
# ValueError if two collectors with the same name are registered to the
# same registry — that would happen every time a MetricConsumer or
# IngestionWorker is constructed in tests. Module-level singletons are
# registered exactly once at import time, which is the prometheus_client
# recommended pattern.
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

# I'm not labelling DLQ events by stage_id because the whole reason a message
# lands in the DLQ is that we couldn't deserialise it — we have no stage_id
# to label with. A stage_id="unknown" label would look meaningful but carry
# no actual information, which is worse than no label at all.
DLQ_EVENTS = prom.Counter(
    "ingestion_dlq_events_total",
    "Total records routed to the DLQ due to deserialisation failure.",
)

WRITE_LATENCY = prom.Histogram(
    "ingestion_write_latency_seconds",
    "Wall time for a single PostgreSQL batch INSERT, in seconds.",
    # I'm using sub-10ms buckets at the low end because our target p99 for
    # a 500-record batch is under 50ms. The default prometheus_client buckets
    # top out at 10s which would compress all our normal traffic into the first
    # two buckets, making the histogram useless for latency percentile estimation.
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
    I'm configuring structlog with JSON output rather than the default
    key=value format because the ingestion layer runs in a container whose
    stdout is scraped by a log aggregator (Loki, CloudWatch, etc.). JSON
    lets the aggregator parse fields without a custom regex, which means
    trace_id and stage_id become filterable dimensions automatically.

    I'm using PrintLoggerFactory (stdout) rather than stdlib logging because
    the ingestion process is a single-purpose worker — there is no existing
    logging hierarchy to integrate with, and the stdlib bridge adds latency
    from the extra handler dispatch for no benefit.
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
    I'm suppressing the per-request access log that WSGIRequestHandler writes
    to stderr by default. In production the scrape happens every 15 seconds —
    that's 4 log lines per minute of pure noise that drowns out real events.
    In tests it generates output that clutters the pytest capture buffer.
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class MetricsServer:
    """
    I'm using wsgiref.simple_server rather than prometheus_client.start_http_server
    because start_http_server's return type changed between prometheus_client
    0.16 and 0.20 (from None to a tuple), making it impossible to call
    shutdown() portably. wsgiref is stdlib and gives us clean lifecycle control:
    start() begins serving in a daemon thread, stop() calls httpd.shutdown()
    synchronously so tests can tear down without a sleep.

    I'm binding to 127.0.0.1 rather than 0.0.0.0 because the /metrics
    endpoint should only be reachable by a local Prometheus sidecar or test
    process. Binding to all interfaces on a container would expose internal
    counters to whatever else is on the network.
    """

    def __init__(self, port: int = 0) -> None:
        # port=0 lets the OS assign a free port — critical for tests running
        # in parallel so they don't collide on a hardcoded port number.
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
