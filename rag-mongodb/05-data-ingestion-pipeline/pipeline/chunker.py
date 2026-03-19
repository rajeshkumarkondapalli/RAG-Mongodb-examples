"""
Chunking Strategy Module
------------------------
Splits an EnrichedDocument into overlapping chunks using one of four strategies:

  1. fixed_size   – split on token/char count with optional overlap
  2. sentence     – split on sentence boundaries
  3. recursive    – LangChain-style recursive character splitting
  4. semantic     – group sentences whose embeddings are similar (cosine similarity
                    threshold); requires an embedding function to be provided.

Each chunk carries a copy of the parent document metadata plus chunk-level
fields: chunk_index, chunk_total, strategy, char_start, char_end.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .enricher import EnrichedDocument


class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass
class ChunkConfig:
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512          # characters (or tokens if tokeniser provided)
    chunk_overlap: int = 64        # characters of overlap between adjacent chunks
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    semantic_threshold: float = 0.75   # cosine similarity threshold for semantic chunking
    min_chunk_length: int = 30         # discard chunks shorter than this


@dataclass
class Chunk:
    content: str
    chunk_index: int
    chunk_total: int          # filled in after all chunks are produced
    strategy: str
    char_start: int
    char_end: int
    metadata: dict[str, Any]
    doc_id: str | None = None  # set after MongoDB upsert


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _fixed_size_split(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Return list of (chunk_text, char_start, char_end)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append((text[start:end], start, end))
        start += size - overlap
    return chunks


def _sentence_split(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Group sentences into chunks up to chunk_size characters."""
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)
    # Recover character positions
    positions: list[tuple[str, int]] = []
    cursor = 0
    for sent in sentences:
        positions.append((sent, cursor))
        cursor += len(sent) + 1  # +1 for the space that was split on

    chunks: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_start = 0
    current_len = 0

    for sent, pos in positions:
        if current_len + len(sent) > chunk_size and current:
            joined = " ".join(current)
            chunks.append((joined, current_start, current_start + len(joined)))
            # Overlap: keep last sentence(s) that fit within overlap chars
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sents
            current_start = pos - overlap_len
            current_len = overlap_len

        current.append(sent)
        if not current_len:
            current_start = pos
        current_len += len(sent)

    if current:
        joined = " ".join(current)
        chunks.append((joined, current_start, current_start + len(joined)))

    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[tuple[str, int, int]]:
    """LangChain-style recursive character text splitter."""

    def _split(txt: str, seps: list[str], offset: int) -> list[tuple[str, int, int]]:
        if not seps or len(txt) <= chunk_size:
            return [(txt, offset, offset + len(txt))]

        sep = seps[0]
        rest = seps[1:]
        parts = txt.split(sep) if sep else list(txt)
        results: list[tuple[str, int, int]] = []
        current = ""
        current_start = offset

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    results.extend(_split(current, rest, current_start))
                current = part
                current_start = offset + txt.find(part, current_start - offset)

        if current:
            results.extend(_split(current, rest, current_start))

        return results

    raw = _split(text, separators, 0)
    # Apply overlap by merging small adjacent chunks and re-splitting
    merged: list[tuple[str, int, int]] = []
    i = 0
    while i < len(raw):
        chunk_text, cstart, cend = raw[i]
        # Try to absorb overlap from the previous chunk
        if merged and overlap:
            prev_text = merged[-1][0]
            tail = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            chunk_text = tail + chunk_text
            cstart = cstart - len(tail)
        merged.append((chunk_text, cstart, cend))
        i += 1

    return merged


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_split(
    text: str,
    embed_fn: Callable[[str], list[float]],
    threshold: float,
    chunk_size: int,
) -> list[tuple[str, int, int]]:
    """
    Group consecutive sentences whose embedding similarity exceeds `threshold`.
    Falls back to sentence-level splitting when no embed_fn is provided.
    """
    sentence_pattern = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_pattern.split(text)

    groups: list[list[str]] = []
    embeddings: list[list[float]] = [embed_fn(s) for s in sentences]

    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        current_len = sum(len(s) for s in current_group)
        if sim >= threshold and current_len + len(sentences[i]) <= chunk_size:
            current_group.append(sentences[i])
        else:
            groups.append(current_group)
            current_group = [sentences[i]]
    groups.append(current_group)

    chunks: list[tuple[str, int, int]] = []
    cursor = 0
    for grp in groups:
        joined = " ".join(grp)
        chunks.append((joined, cursor, cursor + len(joined)))
        cursor += len(joined) + 1

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    doc: EnrichedDocument,
    config: ChunkConfig | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[Chunk]:
    """
    Split an EnrichedDocument into Chunk objects.

    Parameters
    ----------
    doc       : EnrichedDocument from enricher.enrich_document()
    config    : ChunkConfig controlling strategy and sizes
    embed_fn  : Required only for ChunkStrategy.SEMANTIC
    """
    config = config or ChunkConfig()
    text = doc.content

    if config.strategy == ChunkStrategy.FIXED_SIZE:
        raw = _fixed_size_split(text, config.chunk_size, config.chunk_overlap)
    elif config.strategy == ChunkStrategy.SENTENCE:
        raw = _sentence_split(text, config.chunk_size, config.chunk_overlap)
    elif config.strategy == ChunkStrategy.RECURSIVE:
        raw = _recursive_split(text, config.separators, config.chunk_size, config.chunk_overlap)
    elif config.strategy == ChunkStrategy.SEMANTIC:
        if embed_fn is None:
            raise ValueError("embed_fn must be provided for semantic chunking strategy")
        raw = _semantic_split(text, embed_fn, config.semantic_threshold, config.chunk_size)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")

    # Filter too-short chunks
    raw = [(t, s, e) for t, s, e in raw if len(t.strip()) >= config.min_chunk_length]

    chunks: list[Chunk] = []
    for idx, (chunk_text, char_start, char_end) in enumerate(raw):
        chunks.append(
            Chunk(
                content=chunk_text.strip(),
                chunk_index=idx,
                chunk_total=0,   # back-filled below
                strategy=config.strategy.value,
                char_start=char_start,
                char_end=char_end,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                    "char_start": char_start,
                    "char_end": char_end,
                    "chunk_strategy": config.strategy.value,
                },
            )
        )

    total = len(chunks)
    for chunk in chunks:
        chunk.chunk_total = total
        chunk.metadata["chunk_total"] = total

    return chunks


def chunk_batch(
    docs: list[EnrichedDocument],
    config: ChunkConfig | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[Chunk]:
    """Chunk a list of EnrichedDocuments, returning a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, config, embed_fn))
    return all_chunks
