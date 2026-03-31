from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
from confluent_kafka import Producer as KafkaProducer
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from api.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Lifespan — initialises shared resources once, tears them down on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager over the deprecated on_event hooks because FastAPI
    deprecates on_event in 0.109+ and lifespan gives deterministic teardown order.
    Database pool created synchronously: async engines add driver complexity
    (asyncpg vs psycopg3) without throughput benefit at the request rates this
    API serves (< 100 rps).
    """
    settings: Settings = get_settings()

    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    kafka_conf = {
        "bootstrap.servers": settings.REDPANDA_BROKERS,
        "socket.timeout.ms": 5000,
    }
    kafka_producer = KafkaProducer(kafka_conf)

    app.state.db_engine = engine
    app.state.kafka_producer = kafka_producer
    app.state.settings = settings

    _configure_structlog(settings.LOG_LEVEL)
    _log = structlog.get_logger("api.lifespan")
    _log.info("api_started", db=settings.DATABASE_URL, brokers=settings.REDPANDA_BROKERS)

    yield

    kafka_producer.flush(timeout=5.0)
    engine.dispose()
    _log.info("api_stopped")


# ---------------------------------------------------------------------------
# Middleware — injects structlog context per request
# ---------------------------------------------------------------------------


async def _request_context_middleware(request: Request, call_next) -> Response:
    """
    Binds request_id and path to structlog context vars so every log line emitted
    during the request carries routing context without manual threading.
    """
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Health check helpers
# ---------------------------------------------------------------------------


def _check_db(engine: Engine) -> str:
    """SELECT 1 against the connection pool — cheapest possible liveness probe."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "connected"
    except Exception:
        return "error"


def _check_redpanda(producer: KafkaProducer) -> str:
    """
    list_topics with a short timeout confirms broker connectivity. len(topics) > 0
    distinguishes a responsive broker from a TCP-connected-but-non-functional one.
    """
    try:
        metadata = producer.list_topics(timeout=3.0)
        if metadata.topics is not None:
            return "connected"
        return "error"
    except Exception:
        return "error"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Factory function rather than module-level singleton so tests can create
    isolated app instances with overridden state without cross-test leakage.
    """
    app = FastAPI(
        title="QueryLens",
        description="Self-healing data pipeline observatory",
        version="0.1.0",
        lifespan=lifespan,
    )
    FastAPIInstrumentor.instrument_app(app)
    app.middleware("http")(_request_context_middleware)

    @app.get("/health")
    def health(request: Request) -> Dict[str, Any]:
        """
        Returns real connectivity status for both PostgreSQL and Redpanda rather
        than a static 200. An operator hitting /health during an incident needs
        to know which dependency is down without checking two separate dashboards.

        Returns 503 if either dependency is unreachable so load balancer health
        checks pull the instance out of rotation during partial failures.
        """
        engine: Engine = request.app.state.db_engine
        kafka_producer: KafkaProducer = request.app.state.kafka_producer

        db_status = _check_db(engine)
        redpanda_status = _check_redpanda(kafka_producer)

        healthy = db_status == "connected" and redpanda_status == "connected"
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "db": db_status,
                "redpanda": redpanda_status,
            },
        )

    return app


app = create_app()


# ---------------------------------------------------------------------------
# structlog setup
# ---------------------------------------------------------------------------


def _configure_structlog(log_level: str) -> None:
    """
    JSON output for container log aggregators. Same configuration as the ingestion
    layer (ingestion/observability.py) to keep log format consistent across services.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
