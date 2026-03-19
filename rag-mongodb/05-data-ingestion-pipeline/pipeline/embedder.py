"""
Embedding Generator Module
--------------------------
Generates dense vector embeddings for Chunk objects using the OpenAI
Embeddings API (text-embedding-3-small by default).

Features
--------
- Batched API calls (up to 2048 items per request)
- Automatic retry with exponential backoff on rate-limit / transient errors
- Embedding dimension validation
- Optional local cache (dict) to avoid re-embedding identical content
- Support for pluggable embedding backends via EmbedBackend protocol
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI, RateLimitError, APIError

from .chunker import Chunk


# ---------------------------------------------------------------------------
# Backend protocol – lets you swap in HuggingFace, Cohere, etc.
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbedBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


@dataclass
class OpenAIEmbedBackend:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536          # 1536 for small, 3072 for large
    max_retries: int = 5
    initial_backoff: float = 1.0    # seconds
    _client: OpenAI = field(default_factory=OpenAI, init=False, repr=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API with retry logic."""
        backoff = self.initial_backoff
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {"input": texts, "model": self.model}
                if self.dimensions and "3-small" in self.model or "3-large" in self.model:
                    kwargs["dimensions"] = self.dimensions
                response = self._client.embeddings.create(**kwargs)
                return [item.embedding for item in response.data]
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2
            except APIError as exc:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError("Embedding failed after max retries")  # unreachable


# ---------------------------------------------------------------------------
# Embedding config
# ---------------------------------------------------------------------------

@dataclass
class EmbedConfig:
    batch_size: int = 256           # chunks per API call
    use_cache: bool = True          # cache embeddings by content hash


# ---------------------------------------------------------------------------
# Embedded chunk
# ---------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    content: str
    embedding: list[float]
    chunk_index: int
    chunk_total: int
    strategy: str
    char_start: int
    char_end: int
    metadata: dict[str, Any]
    doc_id: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[Chunk],
    backend: EmbedBackend | None = None,
    config: EmbedConfig | None = None,
) -> list[EmbeddedChunk]:
    """
    Generate embeddings for a list of Chunk objects.

    Parameters
    ----------
    chunks  : list of Chunk from chunker.chunk_document()
    backend : EmbedBackend implementation (defaults to OpenAIEmbedBackend)
    config  : EmbedConfig controlling batch size and caching
    """
    backend = backend or OpenAIEmbedBackend()
    config = config or EmbedConfig()

    cache: dict[str, list[float]] = {}

    def _content_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # Identify which chunks need fresh embeddings
    to_embed_indices: list[int] = []
    to_embed_texts: list[str] = []

    for i, chunk in enumerate(chunks):
        key = _content_key(chunk.content)
        if config.use_cache and key in cache:
            continue
        to_embed_indices.append(i)
        to_embed_texts.append(chunk.content)

    # Batched embedding calls
    all_embeddings: list[list[float]] = []
    for batch_start in range(0, len(to_embed_texts), config.batch_size):
        batch = to_embed_texts[batch_start: batch_start + config.batch_size]
        all_embeddings.extend(backend.embed(batch))

    # Populate cache
    for text, embedding in zip(to_embed_texts, all_embeddings):
        cache[_content_key(text)] = embedding

    # Build EmbeddedChunk list
    result: list[EmbeddedChunk] = []
    for chunk in chunks:
        key = _content_key(chunk.content)
        result.append(
            EmbeddedChunk(
                content=chunk.content,
                embedding=cache[key],
                chunk_index=chunk.chunk_index,
                chunk_total=chunk.chunk_total,
                strategy=chunk.strategy,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                metadata=chunk.metadata,
            )
        )

    return result
