"""
Similarity Metrics Module
--------------------------
Pure-Python similarity functions (no external ML dependencies).

Supported metrics
-----------------
  cosine      – angle-based; standard for most embedding models
  euclidean   – L2 distance (converted to a [0,1] similarity score)
  dot_product – inner product; works well with unit-norm embeddings
  manhattan   – L1 distance similarity
  jaccard     – set-based; useful for sparse / binary vectors

Also provides:
  - vector_search_pipeline()  – build a MongoDB $vectorSearch aggregation pipeline
  - hybrid_search_pipeline()  – combine $vectorSearch + $search (BM25) via RRF
  - find_similar_chunks()     – run vector search against a live collection
"""

import math
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymongo.collection import Collection


# ---------------------------------------------------------------------------
# Metric enum
# ---------------------------------------------------------------------------

class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"


# ---------------------------------------------------------------------------
# Pure-Python implementations
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def euclidean_similarity(a: list[float], b: list[float]) -> float:
    """Convert L2 distance to a [0, 1] similarity score via 1 / (1 + dist)."""
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return 1.0 / (1.0 + dist)


def dot_product_similarity(a: list[float], b: list[float]) -> float:
    """Raw dot product (assumes vectors are already normalised)."""
    return sum(x * y for x, y in zip(a, b))


def manhattan_similarity(a: list[float], b: list[float]) -> float:
    """Convert L1 distance to a [0, 1] similarity score."""
    dist = sum(abs(x - y) for x, y in zip(a, b))
    return 1.0 / (1.0 + dist)


def jaccard_similarity(a: list[float], b: list[float], threshold: float = 0.5) -> float:
    """
    Binary Jaccard similarity after thresholding to a sparse set.
    Treat each dimension as active if its value >= threshold.
    """
    set_a = {i for i, v in enumerate(a) if v >= threshold}
    set_b = {i for i, v in enumerate(b) if v >= threshold}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def compute_similarity(
    a: list[float],
    b: list[float],
    metric: SimilarityMetric = SimilarityMetric.COSINE,
) -> float:
    """Dispatch to the requested similarity function."""
    if metric == SimilarityMetric.COSINE:
        return cosine_similarity(a, b)
    elif metric == SimilarityMetric.EUCLIDEAN:
        return euclidean_similarity(a, b)
    elif metric == SimilarityMetric.DOT_PRODUCT:
        return dot_product_similarity(a, b)
    elif metric == SimilarityMetric.MANHATTAN:
        return manhattan_similarity(a, b)
    elif metric == SimilarityMetric.JACCARD:
        return jaccard_similarity(a, b)
    raise ValueError(f"Unknown metric: {metric}")


def rank_by_similarity(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
    embedding_field: str = "embedding",
    metric: SimilarityMetric = SimilarityMetric.COSINE,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Rank candidate documents by similarity to a query embedding.

    Each candidate dict must contain `embedding_field`.
    Returns top-k candidates sorted by descending similarity score,
    each augmented with a `_score` key.
    """
    scored = []
    for doc in candidates:
        vec = doc.get(embedding_field)
        if vec is None:
            continue
        score = compute_similarity(query_embedding, vec, metric)
        scored.append({**doc, "_score": score})
    return sorted(scored, key=lambda x: x["_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# MongoDB aggregation pipeline builders
# ---------------------------------------------------------------------------

def vector_search_pipeline(
    query_embedding: list[float],
    index_name: str = "vector_index",
    embedding_field: str = "embedding",
    num_candidates: int = 150,
    top_k: int = 10,
    pre_filter: dict[str, Any] | None = None,
    project_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a MongoDB $vectorSearch aggregation pipeline.

    Parameters
    ----------
    query_embedding : dense query vector
    index_name      : Atlas vector search index name
    num_candidates  : candidate pool before re-ranking (>= top_k)
    top_k           : final number of results
    pre_filter      : optional MQL filter applied before ANN search
    project_fields  : fields to return (embedding excluded by default)
    """
    vector_stage: dict[str, Any] = {
        "$vectorSearch": {
            "index": index_name,
            "path": embedding_field,
            "queryVector": query_embedding,
            "numCandidates": num_candidates,
            "limit": top_k,
        }
    }
    if pre_filter:
        vector_stage["$vectorSearch"]["filter"] = pre_filter

    project: dict[str, Any] = {"embedding": 0}  # always exclude raw embedding
    if project_fields:
        project = {f: 1 for f in project_fields}
        project["_id"] = 1

    return [
        vector_stage,
        {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        {"$project": project},
    ]


def hybrid_search_pipeline(
    query_embedding: list[float],
    query_text: str,
    vector_index: str = "vector_index",
    text_index: str = "text_index",
    embedding_field: str = "embedding",
    num_candidates: int = 150,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) hybrid pipeline combining vector + BM25 results.

    Uses $unionWith to merge two search results then applies RRF scoring.
    """
    return [
        # Branch 1: vector search
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_field,
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": top_k,
            }
        },
        {"$addFields": {"_vector_rank": {"$meta": "vectorSearchScore"}, "_id_str": {"$toString": "$_id"}}},
        {"$project": {"embedding": 0}},
        # Merge with Branch 2: full-text search
        {
            "$unionWith": {
                "coll": "chunks",   # same collection
                "pipeline": [
                    {
                        "$search": {
                            "index": text_index,
                            "text": {"query": query_text, "path": "content"},
                        }
                    },
                    {"$limit": top_k},
                    {"$addFields": {"_text_score": {"$meta": "searchScore"}, "_id_str": {"$toString": "$_id"}}},
                    {"$project": {"embedding": 0}},
                ],
            }
        },
        # RRF merge
        {
            "$group": {
                "_id": "$_id_str",
                "doc": {"$first": "$$ROOT"},
                "vector_rank": {"$max": "$_vector_rank"},
                "text_score": {"$max": "$_text_score"},
            }
        },
        {
            "$addFields": {
                "_rrf_score": {
                    "$add": [
                        {"$divide": [1.0, {"$add": [rrf_k, {"$ifNull": ["$vector_rank", 0]}]}]},
                        {"$divide": [1.0, {"$add": [rrf_k, {"$ifNull": ["$text_score", 0]}]}]},
                    ]
                }
            }
        },
        {"$sort": {"_rrf_score": -1}},
        {"$limit": top_k},
        {"$replaceRoot": {"newRoot": {"$mergeObjects": ["$doc", {"_rrf_score": "$_rrf_score"}]}}},
    ]


# ---------------------------------------------------------------------------
# Live collection queries
# ---------------------------------------------------------------------------

def find_similar_chunks(
    query_embedding: list[float],
    collection: "Collection",
    index_name: str = "vector_index",
    top_k: int = 10,
    pre_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run vector search against a MongoDB Atlas collection."""
    pipeline = vector_search_pipeline(
        query_embedding,
        index_name=index_name,
        num_candidates=top_k * 15,
        top_k=top_k,
        pre_filter=pre_filter,
    )
    return list(collection.aggregate(pipeline))
