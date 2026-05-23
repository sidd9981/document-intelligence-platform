"""
Central configuration for the application.

All environment variables are read and validated here. Every other module
imports from this file. No other module calls os.getenv() directly.

Pydantic Settings validates all fields at import time. If a required
variable is missing or has the wrong type, the application fails
immediately with a descriptive error rather than failing silently
during a request.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """Connection settings for PostgreSQL.

    Stores document metadata, tenant configurations, audit logs,
    cost logs, entity registry, and faithfulness failure records.
    """

    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    db: str = Field(default="finsight", alias="POSTGRES_DB")
    user: str = Field(default="finsight", alias="POSTGRES_USER")
    password: str = Field(default="changeme", alias="POSTGRES_PASSWORD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def dsn(self) -> str:
        """Async connection string for asyncpg."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class QdrantSettings(BaseSettings):
    """Connection settings for Qdrant vector store.

    Qdrant holds both dense vectors (semantic similarity) and sparse
    vectors (keyword matching). Both collections are queried on every
    retrieval request and their results fused via RRF.
    """

    host: str = Field(default="localhost", alias="QDRANT_HOST")
    port: int = Field(default=6333, alias="QDRANT_PORT")
    collection_dense: str = Field(
        default="filings_dense",
        alias="QDRANT_COLLECTION_DENSE",
    )
    collection_sparse: str = Field(
        default="filings_sparse",
        alias="QDRANT_COLLECTION_SPARSE",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class RedisSettings(BaseSettings):
    """Connection settings for Redis.

    Redis serves four distinct purposes in this system, each using a
    separate logical database to prevent key collisions:

    - DB 0: semantic query cache (avoid redundant LLM calls)
    - DB 1: token budget counters (per-tenant daily spend tracking)
    - DB 2: priority queues (weighted fair scheduling across tenants)
    - DB 3: ingestion streams (Redis Streams as local Kafka substitute)
    """

    url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    cache_db: int = Field(default=0, alias="REDIS_CACHE_DB")
    budget_db: int = Field(default=1, alias="REDIS_BUDGET_DB")
    queue_db: int = Field(default=2, alias="REDIS_QUEUE_DB")
    streams_db: int = Field(default=3, alias="REDIS_STREAMS_DB")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OllamaSettings(BaseSettings):
    """Settings for the local LLM and embedding model via Ollama.

    Ollama exposes an OpenAI-compatible API. The application uses the
    OpenAI Python client pointed at the Ollama base URL. Swapping to
    vLLM or the Anthropic API in production requires changing only
    the base URL and model name, not any application logic.

    keep_alive controls how long Ollama holds the model in memory after
    a request. Set to 0 in development to reclaim RAM immediately after
    each inference call. On a machine with 18GB unified memory shared
    with the OS and other services, holding the model in memory
    continuously would cause memory pressure.
    """

    base_url: str = Field(
        default="http://localhost:11434/v1",
        alias="OLLAMA_BASE_URL",
    )
    model: str = Field(default="llama3.2:3b", alias="OLLAMA_MODEL")
    keep_alive: int = Field(default=0, alias="OLLAMA_KEEP_ALIVE")
    embedding_model: str = Field(
        default="nomic-embed-text",
        alias="EMBEDDING_MODEL",
    )
    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OtelSettings(BaseSettings):
    """Settings for OpenTelemetry tracing.

    The collector endpoint receives spans from all services and routes
    them to the configured backend. In Phase 1 the backend is the
    collector's debug exporter (prints to logs). In Phase 6 it routes
    to Langfuse and Grafana Tempo.
    """

    endpoint: str = Field(
        default="http://localhost:4317",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    service_name: str = Field(
        default="finsight",
        alias="OTEL_SERVICE_NAME",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppSettings(BaseSettings):
    """General application settings."""

    env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings:
    """Root settings object. Import and use this everywhere.

    Instantiated once at module load time. All sub-settings are
    validated at that point. If any required variable is missing
    or invalid, the import fails with a descriptive error.

    Usage:
        from finsight.config.settings import settings

        print(settings.qdrant.host)
        print(settings.ollama.model)
        print(settings.postgres.dsn)
    """

    def __init__(self) -> None:
        self.postgres = PostgresSettings()
        self.qdrant = QdrantSettings()
        self.redis = RedisSettings()
        self.ollama = OllamaSettings()
        self.otel = OtelSettings()
        self.app = AppSettings()


settings = Settings()