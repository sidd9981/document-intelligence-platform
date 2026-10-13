"""
LLM and embedding client wrapper.

All calls to Ollama go through this module. Direct use of the OpenAI
client outside this file is not permitted.

Ollama exposes an OpenAI-compatible API. We use the official OpenAI
Python client pointed at the Ollama base URL, so swapping to vLLM or
another OpenAI-compatible server means changing the base URL and model
names in settings, not application code.

Tenants are configured with model tiers rather than model names.
MODEL_TIERS maps each tier to the Ollama model that serves it, and
resolve_model applies that mapping.

Responsibilities:
    - Token counting before every completion call so the caller knows
      the cost before committing to the request.
    - OTEL spans on every operation with model, token counts and latency.

OpenAI client exceptions propagate to the caller. Agents catch them and
return structured errors.
"""

import time
from collections.abc import AsyncGenerator

import tiktoken
from openai import AsyncOpenAI

from finsight.config.settings import settings
from finsight.telemetry.tracing import get_tracer

tracer = get_tracer(__name__)

MODEL_TIERS = {
    "small": settings.ollama.model,
    "medium": "llama3.1:8b",
}

_client: AsyncOpenAI | None = None
_tokenizer: tiktoken.Encoding | None = None


def resolve_model(tier_or_name: str) -> str:
    """Map a tenant model tier to an Ollama model name.

    Tier names come from tenant_configs.allowed_models. A value that is not
    a known tier is treated as a literal Ollama model name, so a tenant can
    also be pinned to one specific model.
    """
    return MODEL_TIERS.get(tier_or_name, tier_or_name)


def get_client() -> AsyncOpenAI:
    """Return the shared OpenAI-compatible client instance.

    Raises:
        RuntimeError: If called before init_client() has been awaited.
    """
    if _client is None:
        raise RuntimeError(
            "llm client is not initialized. "
            "call init_client() during application startup."
        )
    return _client


def get_tokenizer() -> tiktoken.Encoding:
    """Return the shared tokenizer instance.

    Uses cl100k_base, the GPT-4 tokenizer. Llama tokenizers differ, so
    counts are approximate, close enough for budget enforcement but not
    exact.

    Raises:
        RuntimeError: If called before init_client() has been awaited.
    """
    if _tokenizer is None:
        raise RuntimeError(
            "tokenizer is not initialized. "
            "call init_client() during application startup."
        )
    return _tokenizer


async def init_client() -> None:
    """Initialize the shared LLM client and tokenizer.

    Must be called once at application startup. Safe to call multiple
    times, subsequent calls return immediately if already initialized.
    """
    global _client, _tokenizer

    if _client is not None:
        return

    with tracer.start_as_current_span("llm.init_client") as span:
        span.set_attribute("llm.base_url", settings.ollama.base_url)
        span.set_attribute("llm.model", settings.ollama.model)

        _client = AsyncOpenAI(
            base_url=settings.ollama.base_url,
            api_key="ollama",
        )

        _tokenizer = tiktoken.get_encoding("cl100k_base")


async def close_client() -> None:
    """Close the LLM client.

    Must be called at application shutdown.
    """
    global _client, _tokenizer

    if _client is None:
        return

    await _client.close()
    _client = None
    _tokenizer = None


def count_tokens(text: str) -> int:
    """Count the approximate number of tokens in a string.

    Used before LLM calls to estimate cost and enforce context window
    limits.

    Args:
        text: The text to tokenize.

    Returns:
        Approximate token count.
    """
    return len(get_tokenizer().encode(text))


async def embed(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Used during ingestion to embed document chunks and during retrieval
    to embed incoming queries. Both use the same model so query vectors
    and chunk vectors are in the same space.

    Args:
        text: A single chunk or query, not a concatenation of several.

    Returns:
        The embedding vector. Length matches settings.ollama.embedding_dim.
    """
    client = get_client()

    with tracer.start_as_current_span("llm.embed") as span:
        span.set_attribute("model", settings.ollama.embedding_model)
        span.set_attribute("text_length", len(text))

        start = time.perf_counter()

        response = await client.embeddings.create(
            model=settings.ollama.embedding_model,
            input=text,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("latency_ms", round(latency_ms, 2))

        return response.data[0].embedding


async def complete(
    prompt: str,
    system: str,
    max_tokens: int,
    model: str | None = None,
    temperature: float = 0.1,
) -> tuple[str, int, int]:
    """Generate a completion for the given prompt.

    Args:
        prompt: The user message content.
        system: The system message that sets context and behavior.
        max_tokens: Maximum tokens to generate in the response.
        model: Ollama model name or tenant tier, resolved through
               resolve_model. Defaults to settings.ollama.model.
        temperature: Sampling temperature. Judges should pass 0.0.

    Returns:
        A tuple of (answer, prompt_tokens, completion_tokens).
    """
    client = get_client()
    model_name = resolve_model(model) if model else settings.ollama.model

    with tracer.start_as_current_span("llm.complete") as span:
        span.set_attribute("model", model_name)
        span.set_attribute("max_tokens", max_tokens)
        span.set_attribute("temperature", temperature)

        prompt_token_count = count_tokens(system + prompt)
        span.set_attribute("prompt_tokens_estimate", prompt_token_count)

        start = time.perf_counter()

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        span.set_attribute("latency_ms", round(latency_ms, 2))

        answer = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else prompt_token_count
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        span.set_attribute("prompt_tokens_actual", prompt_tokens)
        span.set_attribute("completion_tokens", completion_tokens)

        return answer, prompt_tokens, completion_tokens


async def stream_complete(
    prompt: str,
    system: str,
    max_tokens: int,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream completion tokens as they are generated.

    Yields each token string as it arrives from Ollama. The caller is
    responsible for assembling the full answer if needed.

    Args:
        prompt: The user message content.
        system: The system message.
        max_tokens: Maximum tokens to generate.
        model: Ollama model name or tenant tier. Defaults to settings.ollama.model.

    Yields:
        Token strings as they arrive.
    """
    client = get_client()
    model_name = resolve_model(model) if model else settings.ollama.model

    with tracer.start_as_current_span("llm.stream_complete") as span:
        span.set_attribute("model", model_name)
        span.set_attribute("max_tokens", max_tokens)

        stream = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token