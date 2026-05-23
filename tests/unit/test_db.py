"""
Unit tests for the database connection pool module.

These tests do not require a running Postgres instance. They verify
the module's behavior in isolation using only its public interface.
"""

import pytest

from finsight.gateway import db


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset the pool state before and after each test.

    The pool module uses a module-level variable. Without resetting it
    between tests, state leaks from one test to the next and tests
    become order-dependent, which makes failures hard to diagnose.
    """
    db._pool = None
    yield
    db._pool = None


def test_get_pool_raises_before_init():
    """get_pool() must raise RuntimeError if called before init_pool().

    The error message must tell the caller what went wrong and how to
    fix it, not just that something failed.
    """
    with pytest.raises(RuntimeError, match="not initialized"):
        db.get_pool()


def test_get_pool_raises_after_close():
    """get_pool() must raise RuntimeError after the pool is closed.

    Simulates the state after close_pool() sets _pool back to None.
    """
    db._pool = None

    with pytest.raises(RuntimeError, match="not initialized"):
        db.get_pool()