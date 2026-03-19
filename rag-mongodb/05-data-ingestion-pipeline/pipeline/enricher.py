"""
Metadata Enrichment Module
--------------------------
Adds derived / inferred metadata to a CleanedDocument before it is chunked:
  - word_count, char_count, sentence_count
  - language detection (lightweight heuristic; swap for langdetect if needed)
  - keyword extraction (top-N TF-style terms)
  - readability tier (simple Flesch proxy)
  - ingest timestamps
  - pipeline version tag
"""

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .cleaner import CleanedDocument

PIPELINE_VERSION = "1.0.0"

# Very small set of English stop-words sufficient for keyword extraction
_STOP_WORDS: frozenset[str] = frozenset(
    """a an the and or but in on at to for of with is are was were be been
    being have has had do does did will would could should may might shall
    this that these those it its itself they them their there here by from
    as if not no so such when where which who whom what how all any both
    each few more most other some than then too very just about above after
    again against also before between during each even here into many much
    now only out over same since still through under until up very while""".split()
)


@dataclass
class EnrichedDocument:
    content: str
    content_hash: str
    metadata: dict[str, Any]


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def _top_keywords(tokens: list[str], n: int = 10) -> list[str]:
    filtered = [t for t in tokens if t not in _STOP_WORDS]
    return [word for word, _ in Counter(filtered).most_common(n)]


def _readability_tier(words: list[str], sentences: int) -> str:
    """
    Very rough Flesch-Kincaid proxy using average word length as a syllable proxy.
    Returns 'simple' | 'intermediate' | 'advanced'.
    """
    if not words:
        return "unknown"
    avg_word_len = sum(len(w) for w in words) / len(words)
    avg_sent_len = len(words) / sentences
    score = avg_word_len + avg_sent_len * 0.1
    if score < 6:
        return "simple"
    elif score < 9:
        return "intermediate"
    return "advanced"


def _detect_language(text: str) -> str:
    """
    Heuristic language tag — checks for common non-ASCII script ranges.
    Falls back to 'en' for Latin text.  Replace with langdetect for accuracy.
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
        return "ja"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    if re.search(r"[\u0400-\u04ff]", text):
        return "ru"
    return "en"


def enrich_document(
    doc: CleanedDocument,
    extra_metadata: dict[str, Any] | None = None,
    top_keywords: int = 10,
) -> EnrichedDocument:
    """
    Attach computed and user-supplied metadata to a CleanedDocument.

    Parameters
    ----------
    doc : CleanedDocument
        Output from cleaner.clean_document().
    extra_metadata : dict, optional
        Additional key-value pairs to merge (e.g. source URL, author, doc_id).
    top_keywords : int
        Number of top keywords to extract.
    """
    tokens = _word_tokens(doc.content)
    sentences = _sentence_count(doc.content)

    computed = {
        # Provenance
        "pipeline_version": PIPELINE_VERSION,
        "ingested_at": int(time.time()),            # Unix epoch seconds
        # Size stats
        "char_count": doc.cleaned_length,
        "word_count": len(tokens),
        "sentence_count": sentences,
        # Language & readability
        "language": _detect_language(doc.content),
        "readability": _readability_tier(tokens, sentences),
        # Keywords (for search faceting / filtering)
        "keywords": _top_keywords(tokens, top_keywords),
        # Change detection
        "content_hash": doc.content_hash,
    }

    merged = {**doc.metadata, **computed, **(extra_metadata or {})}

    return EnrichedDocument(
        content=doc.content,
        content_hash=doc.content_hash,
        metadata=merged,
    )


def enrich_batch(
    docs: list[CleanedDocument],
    extra_metadata_list: list[dict[str, Any]] | None = None,
    top_keywords: int = 10,
) -> list[EnrichedDocument]:
    """Enrich a batch of CleanedDocuments."""
    extras = extra_metadata_list or [{}] * len(docs)
    return [
        enrich_document(doc, extra, top_keywords)
        for doc, extra in zip(docs, extras)
    ]
