"""
Database migration runner.

Applies all pending migrations in infra/migrations/ to the configured
Postgres database. Safe to run multiple times — already applied
migrations are skipped.

Usage:
    python scripts/migrate.py
"""

import sys
from pathlib import Path

from yoyo import read_migrations, get_backend

sys.path.insert(0, str(Path(__file__).parent.parent))

from finsight.config.settings import settings


def main() -> None:
    """Apply all pending migrations to the database."""
    migrations_path = Path(__file__).parent.parent / "infra" / "migrations"

    backend = get_backend(settings.postgres.dsn.replace("postgresql://", "postgresql://"))
    migrations = read_migrations(str(migrations_path))

    with backend.lock():
        backend.apply_migrations(backend.to_apply(migrations))

    print("migrations applied successfully")


if __name__ == "__main__":
    main()