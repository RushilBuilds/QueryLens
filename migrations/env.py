from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# I'm importing Base from ingestion.models so Alembic's autogenerate diff has
# access to the full target metadata. Without this import, `alembic revision
# --autogenerate` would always produce empty migrations because it would compare
# the database schema against an empty MetaData object rather than the declared
# models. Every new model module added in later phases must be imported here too.
from ingestion.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    """
    I'm prioritising DATABASE_URL from the environment over the value in alembic.ini
    so that CI, Docker Compose, and testcontainers-based tests can all inject their
    own connection strings without editing the ini file. The ini value serves as a
    documented default for local development only — it is never read in production or
    during automated tests.
    """
    return os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    """
    I'm keeping offline mode so that the migration can be rendered to SQL and
    reviewed in code review without needing a running database. Useful when the
    DBA wants to inspect the DDL before it touches a shared environment.
    """
    url = _db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": _db_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
