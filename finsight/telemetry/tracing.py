"""
OpenTelemetry tracing setup.

Call setup_tracing() once at application startup before serving any
requests. After that, every module gets a tracer by calling
get_tracer(__name__) and wraps external calls in spans.

Why this exists as a separate module:
    Tracing setup involves global state — registering a provider that
    all subsequent get_tracer() calls resolve against. Isolating that
    here means the rest of the codebase never touches the SDK directly.
    It only imports get_tracer() and uses it. If we ever swap OTEL for
    another tracing system, this is the only file that changes.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from finsight.config.settings import settings


def setup_tracing() -> None:
    """Initialize the global tracer provider.

    Must be called once before the application serves any requests.
    Subsequent calls are safe but have no effect because the global
    provider is already set.

    In development (APP_ENV=development), spans are exported to both
    the OTEL collector and stdout so you can see traces in the terminal
    without opening a separate UI.

    In production, only the OTEL collector exporter is active.
    """
    resource = Resource.create(
        {
            "service.name": settings.otel.service_name,
            "service.version": "0.1.0",
            "deployment.environment": settings.app.env,
        }
    )

    provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.otel.endpoint,
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if settings.app.env == "development":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for the given module.

    Args:
        name: The module name. Always pass __name__ so spans are
              automatically labeled with their source file.

    Returns:
        A tracer instance bound to the given name.

    Usage:
        tracer = get_tracer(__name__)

        async def search_vectors(query: str) -> list[Chunk]:
            with tracer.start_as_current_span("qdrant.vector_search") as span:
                span.set_attribute("query_length", len(query))
                result = await qdrant_client.search(...)
                span.set_attribute("chunks_returned", len(result))
                return result
    """
    return trace.get_tracer(name)