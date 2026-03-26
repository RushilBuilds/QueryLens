from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Base imported from ingestion.models so Alembic's autogenerate diff has access
# to the full target metadata. Without this, --autogenerate always produces empty
# migrations (compares against an empty MetaData). Every new model module must be
# imported here too.
from ingestion.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _db_url() -> str:
    """
    DATABASE_URL from the environment takes priority over alembic.ini so CI,
    Docker Compose, and testcontainers can inject connection strings without
    editing the ini. The ini value is a local-development default only.
    """
    return os.environ.get("DATABASE_URL", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    """
    Offline mode renders the migration to SQL for code review without a running
    database, useful when a DBA needs to inspect DDL before it touches a shared
    environment.
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
