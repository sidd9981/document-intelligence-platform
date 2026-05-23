"""
Postgres connection pool management.

Provides a single shared connection pool for the entire application.
The pool is initialized once at application startup and closed at
shutdown.

All database operations in the application import get_pool() from
here. No other module creates its own connections directly.

Why a single shared pool:
    asyncpg pools are expensive to create and have a fixed size.
    Creating multiple pools wastes connections and complicates
    connection limit management. One pool shared across all
    components gives predictable behavior under load.
"""

import asyncpg

from finsight.config.settings import settings
from finsight.telemetry.tracing import get_tracer

tracer = get_tracer(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    """Initialize the shared connection pool.

    Must be called once at application startup before any database
    operations. Subsequent calls are safe — if the pool already
    exists this function returns immediately.

    Pool sizing:
        min_size=2 keeps two connections warm at all times so the
        first requests after startup do not pay connection overhead.
        max_size=10 caps total connections at 10. With three tenant
        teams and typical query patterns this is sufficient for
        development. Production would size this based on concurrent
        query volume and Postgres max_connections setting.
    """
    global _pool

    if _pool is not None:
        return

    with tracer.start_as_current_span("db.init_pool") as span:
        span.set_attribute("db.host", settings.postgres.host)
        span.set_attribute("db.port", settings.postgres.port)
        span.set_attribute("db.name", settings.postgres.db)

        _pool = await asyncpg.create_pool(
            dsn=settings.postgres.dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )

        span.set_attribute("pool.min_size", 2)
        span.set_attribute("pool.max_size", 10)


async def close_pool() -> None:
    """Close the connection pool gracefully.

    Must be called at application shutdown to release all connections
    back to Postgres. Skips silently if the pool was never initialized.
    """
    global _pool

    if _pool is None:
        return

    await _pool.close()
    _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the shared connection pool.

    Raises:
        RuntimeError: If called before init_pool() has been awaited.
            This indicates a programming error — the pool must be
            initialized during application startup before any request
            handler runs.
    """
    if _pool is None:
        raise RuntimeError(
            "database pool is not initialized. "
            "call init_pool() during application startup."
        )
    return _pool