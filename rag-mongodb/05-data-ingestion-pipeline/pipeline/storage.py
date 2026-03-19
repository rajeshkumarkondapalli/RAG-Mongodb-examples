"""
MongoDB Storage & Indexing Module
----------------------------------
Persists EmbeddedChunks into MongoDB Atlas and manages vector search indexes.

Features
--------
- Upsert by content_hash (idempotent re-ingestion)
- Soft-delete support (deleted_at field)
- Vector search index creation (cosine, euclidean, dotProduct)
- Text search index creation for hybrid search
- Utility queries: list stale docs, fetch by source, count by status
"""

import time
from dataclasses import dataclass
from typing import Any

from pymongo import MongoClient, UpdateOne, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError
from pymongo.operations import SearchIndexModel

from .embedder import EmbeddedChunk


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class StorageConfig:
    uri: str = "mongodb://localhost:27017"
    db_name: str = "rag_pipeline"
    collection_name: str = "chunks"
    vector_index_name: str = "vector_index"
    text_index_name: str = "text_index"
    embedding_dimensions: int = 1536
    similarity_metric: str = "cosine"   # cosine | euclidean | dotProduct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_collection(config: StorageConfig) -> Collection:
    client = MongoClient(config.uri)
    return client[config.db_name][config.collection_name]


def _chunk_to_doc(chunk: EmbeddedChunk) -> dict[str, Any]:
    """Convert EmbeddedChunk to a MongoDB document."""
    return {
        "content": chunk.content,
        "embedding": chunk.embedding,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "strategy": chunk.strategy,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "metadata": chunk.metadata,
        "content_hash": chunk.metadata.get("content_hash", ""),
        # Top-level copies for Atlas filter fields
        "source": chunk.metadata.get("source"),
        "category": chunk.metadata.get("category"),
        "language": chunk.metadata.get("language"),
        "pipeline_version": chunk.metadata.get("pipeline_version"),
        "ingested_at": chunk.metadata.get("ingested_at", int(time.time())),
        "updated_at": int(time.time()),
        "deleted_at": None,
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunks: list[EmbeddedChunk],
    config: StorageConfig,
) -> dict[str, int]:
    """
    Upsert EmbeddedChunks into MongoDB.

    Uses content_hash + chunk_index as the idempotency key so re-running the
    pipeline on unchanged data is a no-op.

    Returns counts: {"upserted": N, "matched": N, "errors": N}
    """
    collection = get_collection(config)
    ops = []
    for chunk in chunks:
        doc = _chunk_to_doc(chunk)
        filter_key = {
            "content_hash": doc["content_hash"],
            "chunk_index": doc["chunk_index"],
        }
        ops.append(
            UpdateOne(
                filter_key,
                {"$set": doc, "$setOnInsert": {"created_at": int(time.time())}},
                upsert=True,
            )
        )

    if not ops:
        return {"upserted": 0, "matched": 0, "errors": 0}

    try:
        result = collection.bulk_write(ops, ordered=False)
        return {
            "upserted": result.upserted_count,
            "matched": result.matched_count,
            "errors": 0,
        }
    except BulkWriteError as bwe:
        errors = len(bwe.details.get("writeErrors", []))
        return {
            "upserted": bwe.details.get("nUpserted", 0),
            "matched": bwe.details.get("nMatched", 0),
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Soft delete
# ---------------------------------------------------------------------------

def soft_delete_by_source(source: str, config: StorageConfig) -> int:
    """Mark all chunks from `source` as deleted. Returns modified count."""
    collection = get_collection(config)
    result = collection.update_many(
        {"source": source, "status": "active"},
        {"$set": {"status": "deleted", "deleted_at": int(time.time())}},
    )
    return result.modified_count


def hard_delete_stale(older_than_days: int, config: StorageConfig) -> int:
    """Permanently remove soft-deleted chunks older than N days."""
    cutoff = int(time.time()) - older_than_days * 86400
    collection = get_collection(config)
    result = collection.delete_many(
        {"status": "deleted", "deleted_at": {"$lt": cutoff}}
    )
    return result.deleted_count


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def create_vector_search_index(config: StorageConfig) -> None:
    """
    Create an Atlas Vector Search index.
    Skipped if the index already exists.
    """
    collection = get_collection(config)
    existing = {idx["name"] for idx in collection.list_search_indexes()}

    if config.vector_index_name not in existing:
        index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": config.embedding_dimensions,
                        "similarity": config.similarity_metric,
                    },
                    {"type": "filter", "path": "source"},
                    {"type": "filter", "path": "category"},
                    {"type": "filter", "path": "language"},
                    {"type": "filter", "path": "status"},
                ]
            },
            name=config.vector_index_name,
            type="vectorSearch",
        )
        collection.create_search_index(model=index_model)
        print(f"[storage] Vector index '{config.vector_index_name}' creation triggered (~60s to build).")
    else:
        print(f"[storage] Vector index '{config.vector_index_name}' already exists.")


def create_text_search_index(config: StorageConfig) -> None:
    """Create a full-text Atlas Search index for hybrid search."""
    collection = get_collection(config)
    existing = {idx["name"] for idx in collection.list_search_indexes()}

    if config.text_index_name not in existing:
        index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "content": {"type": "string", "analyzer": "lucene.standard"},
                        "source": {"type": "string"},
                        "category": {"type": "string"},
                        "metadata.keywords": {"type": "string"},
                    },
                }
            },
            name=config.text_index_name,
            type="search",
        )
        collection.create_search_index(model=index_model)
        print(f"[storage] Text index '{config.text_index_name}' creation triggered.")
    else:
        print(f"[storage] Text index '{config.text_index_name}' already exists.")


def create_operational_indexes(config: StorageConfig) -> None:
    """Create standard MongoDB indexes for operational queries."""
    collection = get_collection(config)
    collection.create_index([("content_hash", ASCENDING), ("chunk_index", ASCENDING)], unique=True)
    collection.create_index([("source", ASCENDING)])
    collection.create_index([("status", ASCENDING)])
    collection.create_index([("ingested_at", DESCENDING)])
    collection.create_index([("updated_at", DESCENDING)])
    print("[storage] Operational indexes created.")


def setup_indexes(config: StorageConfig) -> None:
    """Convenience function: create all indexes at once."""
    create_operational_indexes(config)
    create_vector_search_index(config)
    create_text_search_index(config)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_sources(config: StorageConfig) -> list[str]:
    """Return a distinct list of active sources."""
    collection = get_collection(config)
    return collection.distinct("source", {"status": "active"})


def count_chunks_by_status(config: StorageConfig) -> dict[str, int]:
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    collection = get_collection(config)
    return {doc["_id"]: doc["count"] for doc in collection.aggregate(pipeline)}


def get_chunks_by_source(source: str, config: StorageConfig) -> list[dict]:
    collection = get_collection(config)
    return list(collection.find({"source": source, "status": "active"}, {"embedding": 0}))
