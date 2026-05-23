"""
LLM and embedding client wrapper.

All calls to Ollama go through this module. Direct use of the OpenAI
client outside this file is not permitted.

Ollama exposes an OpenAI-compatible API. We use the official OpenAI
Python client pointed at the Ollama base URL. This means swapping
to vLLM or the Anthropic API in production requires changing only
the base URL and model name in settings — no application code changes.

Responsibilities:
    - Token counting before every completion call so the caller knows
      the cost before committing to the request.
    - OTEL spans on every operation with token counts and latency
      as span attributes.
    - Consistent error handling so raw OpenAI client exceptions do
      not propagate into agent code.
"""

import time

import tiktoken
from openai import AsyncOpenAI

from finsight.config.settings import settings
from finsight.telemetry.tracing import get_tracer
from collections.abc import AsyncGenerator

tracer = get_tracer(__name__)

_client: AsyncOpenAI | None = None
_tokenizer: tiktoken.Encoding | None = None


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

    Uses cl100k_base which is the tokenizer for GPT-4 and is
    compatible with Llama models for approximate token counting.
    Exact token counts vary slightly by model but cl100k_base
    gives a close enough estimate for budget enforcement.

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
    times — subsequent calls return immediately if already initialized.
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
    """Count the number of tokens in a string.

    Used before LLM calls to estimate cost and enforce context
    window limits. Called by the input harness to verify the
    constructed prompt fits within the tenant's max_context_tokens.

    Args:
        text: The text to tokenize.

    Returns:
        Approximate token count.
    """
    return len(get_tokenizer().encode(text))


async def embed(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Used during ingestion to embed document chunks and during
    retrieval to embed incoming queries. Both use the same model
    so query vectors and chunk vectors are in the same space.

    Args:
        text: The text to embed. Should be a single chunk or query,
              not a concatenation of multiple texts.

    Returns:
        A list of floats representing the embedding vector. Length
        matches settings.ollama.embedding_dim.
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
) -> tuple[str, int, int]:
    """Generate a completion for the given prompt.

    Args:
        prompt: The user message content.
        system: The system message that sets context and behavior.
        max_tokens: Maximum tokens to generate in the response.

    Returns:
        A tuple of (answer, prompt_tokens, completion_tokens).
        Token counts are used by the caller to meter actual usage
        against the tenant's daily budget.
    """
    client = get_client()

    with tracer.start_as_current_span("llm.complete") as span:
        span.set_attribute("model", settings.ollama.model)
        span.set_attribute("max_tokens", max_tokens)

        prompt_token_count = count_tokens(system + prompt)
        span.set_attribute("prompt_tokens_estimate", prompt_token_count)

        start = time.perf_counter()

        response = await client.chat.completions.create(
            model=settings.ollama.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
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
) -> AsyncGenerator[str, None]:
    """Stream completion tokens as they are generated.

    Yields each token string as it arrives from Ollama. The caller
    is responsible for assembling the full answer if needed.
    Prompt token count is estimated since streaming responses don't
    always return usage stats mid-stream.

    Args:
        prompt: The user message content.
        system: The system message.
        max_tokens: Maximum tokens to generate.

    Yields:
        Token strings as they arrive.
    """
    client = get_client()

    with tracer.start_as_current_span("llm.stream_complete") as span:
        span.set_attribute("model", settings.ollama.model)
        span.set_attribute("max_tokens", max_tokens)

        stream = await client.chat.completions.create(
            model=settings.ollama.model,
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