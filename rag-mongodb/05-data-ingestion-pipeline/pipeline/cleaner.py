"""
Data Cleaning Module
--------------------
Normalises raw documents before chunking and embedding:
  - Strip / collapse whitespace
  - Remove control characters and non-printable characters
  - Normalise Unicode (NFC)
  - De-duplicate consecutive blank lines
  - Optional: remove boilerplate patterns via configurable regex list
  - Compute a content hash for change-detection
"""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


# Patterns considered "boilerplate" that can optionally be stripped
_DEFAULT_BOILERPLATE: list[str] = [
    r"©\s*\d{4}.*",          # copyright lines
    r"All rights reserved.*",
    r"Page \d+ of \d+",
    r"Confidential.*",
]


@dataclass
class CleaningConfig:
    remove_boilerplate: bool = True
    boilerplate_patterns: list[str] = field(default_factory=lambda: list(_DEFAULT_BOILERPLATE))
    min_content_length: int = 20  # discard documents shorter than this after cleaning


@dataclass
class CleanedDocument:
    content: str
    content_hash: str
    original_length: int
    cleaned_length: int
    metadata: dict[str, Any]
    is_valid: bool
    rejection_reason: str | None = None


def clean_document(raw: dict[str, Any], config: CleaningConfig | None = None) -> CleanedDocument:
    """
    Clean a single raw document dict that must contain at least a 'content' key.

    Returns a CleanedDocument with the sanitised text and validity flag.
    """
    config = config or CleaningConfig()
    content: str = raw.get("content", "") or ""
    metadata: dict[str, Any] = {k: v for k, v in raw.items() if k != "content"}

    original_length = len(content)

    # 1. Unicode normalisation
    content = unicodedata.normalize("NFC", content)

    # 2. Remove control / non-printable characters (keep newlines and tabs)
    content = re.sub(r"[^\S\n\t ]+", " ", content)          # collapse exotic whitespace
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", content)  # strip control chars

    # 3. Remove boilerplate patterns
    if config.remove_boilerplate:
        for pattern in config.boilerplate_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.MULTILINE)

    # 4. Collapse multiple consecutive blank lines → single blank line
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 5. Strip leading / trailing whitespace per line, then overall
    content = "\n".join(line.rstrip() for line in content.splitlines())
    content = content.strip()

    # 6. Compute SHA-256 content hash (used for dedup / change detection)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 7. Validity check
    is_valid = len(content) >= config.min_content_length
    rejection_reason = None if is_valid else f"Content too short after cleaning ({len(content)} chars)"

    return CleanedDocument(
        content=content,
        content_hash=content_hash,
        original_length=original_length,
        cleaned_length=len(content),
        metadata=metadata,
        is_valid=is_valid,
        rejection_reason=rejection_reason,
    )


def clean_batch(
    documents: list[dict[str, Any]],
    config: CleaningConfig | None = None,
) -> tuple[list[CleanedDocument], list[CleanedDocument]]:
    """
    Clean a batch of raw documents.

    Returns (valid_docs, rejected_docs).
    """
    valid, rejected = [], []
    for doc in documents:
        cleaned = clean_document(doc, config)
        (valid if cleaned.is_valid else rejected).append(cleaned)
    return valid, rejected
