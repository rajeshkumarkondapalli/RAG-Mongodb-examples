"""
Batch Ingestion Job
--------------------
End-to-end orchestrator that:
  1. Fetches raw documents (from MongoDB source collection or inline data)
  2. Cleans documents
  3. Enriches metadata
  4. Chunks using configurable strategy
  5. Generates embeddings
  6. Upserts into the destination MongoDB collection
  7. Optionally soft-deletes removed sources
  8. Reports metrics and fires alerts

Usage
-----
  # Full ingest with default config
  python batch_job.py

  # Custom options
  python batch_job.py --strategy sentence --chunk-size 300 --delete-stale

  # Dry-run (skip embedding + storage)
  python batch_job.py --dry-run

Environment variables (.env)
  MONGODB_URI      – Atlas connection string
  OPENAI_API_KEY   – OpenAI API key for embeddings
  DB_NAME          – destination database  (default: rag_pipeline)
  COLLECTION_NAME  – destination collection (default: chunks)
"""

import argparse
import os
import uuid
import time
from dotenv import load_dotenv

from pipeline.cleaner import CleaningConfig, clean_batch
from pipeline.enricher import enrich_batch
from pipeline.chunker import ChunkConfig, ChunkStrategy, chunk_batch
from pipeline.embedder import EmbedConfig, OpenAIEmbedBackend, embed_chunks
from pipeline.storage import StorageConfig, upsert_chunks, setup_indexes, soft_delete_by_source, hard_delete_stale
from monitoring.metrics import (
    PipelineMetrics, RunReport, evaluate_alerts,
    get_logger, stage_timer,
)

load_dotenv()

logger = get_logger("batch_job")


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

def fetch_documents(source_uri: str | None = None) -> list[dict]:
    """
    Fetch raw documents.

    When source_uri is set, reads from a MongoDB collection named 'raw_documents'.
    Otherwise falls back to the sample dataset bundled with the repo.
    """
    if source_uri:
        from pymongo import MongoClient
        client = MongoClient(source_uri)
        db = client[os.getenv("SOURCE_DB", "rag_source")]
        docs = list(client[db.name]["raw_documents"].find({}, {"_id": 0}))
        logger.info(f"Fetched {len(docs)} documents from MongoDB source")
        return docs

    # Built-in sample data
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.sample_data import SAMPLE_DOCUMENTS

    # Extend sample data with a few more documents to showcase variety
    extra = [
        {
            "title": "Embedding Models Comparison",
            "source": "ai_concepts",
            "category": "ai",
            "content": (
                "OpenAI text-embedding-3-small and text-embedding-3-large are state-of-the-art "
                "embedding models that produce 1536 and 3072-dimensional vectors respectively. "
                "They significantly outperform the older ada-002 model on MTEB benchmarks. "
                "Cohere Embed v3 and Voyage AI embeddings are strong alternatives offering "
                "competitive performance at lower cost. Choosing the right model involves "
                "balancing retrieval quality, latency, and cost per token."
            ),
        },
        {
            "title": "Vector Database Indexing Strategies",
            "source": "mongodb_docs",
            "category": "database",
            "content": (
                "Approximate Nearest Neighbour (ANN) algorithms like HNSW (Hierarchical Navigable "
                "Small World) and IVF (Inverted File Index) make vector search scalable to billions "
                "of vectors. HNSW offers better recall-latency trade-offs for most workloads while "
                "IVF-PQ (Product Quantization) reduces memory footprint. MongoDB Atlas uses HNSW "
                "under the hood and supports filtering ANN results by metadata pre-filters to "
                "narrow candidate sets before the nearest-neighbour search."
            ),
        },
        {
            "title": "Data Pipeline Observability",
            "source": "engineering",
            "category": "operations",
            "content": (
                "Observability in data pipelines means capturing the right signals—metrics, logs, "
                "and traces—at each stage. Key metrics include document rejection rate, embedding "
                "latency, storage error rate, and end-to-end throughput (docs/sec). Structured "
                "JSON logs enable downstream parsing by tools like Datadog, Grafana Loki, or "
                "CloudWatch. Alert rules on rejection-rate thresholds help detect upstream data "
                "quality issues before they silently degrade retrieval performance."
            ),
        },
    ]
    return SAMPLE_DOCUMENTS + extra


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_pipeline(
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    storage_config: StorageConfig,
    dry_run: bool,
    delete_stale: bool,
    stale_days: int,
    metrics: PipelineMetrics,
) -> None:

    # ── 1. Fetch ──────────────────────────────────────────────────────────
    with stage_timer("fetch", logger) as ctx:
        raw_docs = fetch_documents(os.getenv("SOURCE_MONGODB_URI"))
        metrics.docs_fetched = len(raw_docs)
        ctx["docs"] = len(raw_docs)

    # ── 2. Clean ─────────────────────────────────────────────────────────
    with stage_timer("clean", logger) as ctx:
        clean_cfg = CleaningConfig()
        valid_docs, rejected_docs = clean_batch(raw_docs, clean_cfg)
        metrics.docs_cleaned = len(valid_docs)
        metrics.docs_rejected = len(rejected_docs)
        ctx["valid"] = len(valid_docs)
        ctx["rejected"] = len(rejected_docs)
        for r in rejected_docs:
            logger.warning("doc_rejected", extra={"reason": r.rejection_reason})

    # ── 3. Enrich ─────────────────────────────────────────────────────────
    with stage_timer("enrich", logger) as ctx:
        enriched_docs = enrich_batch(valid_docs)
        ctx["docs"] = len(enriched_docs)

    # ── 4. Chunk ──────────────────────────────────────────────────────────
    with stage_timer("chunk", logger) as ctx:
        chunk_cfg = ChunkConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = chunk_batch(enriched_docs, chunk_cfg)
        metrics.chunks_created = len(chunks)
        ctx["chunks"] = len(chunks)

    if dry_run:
        logger.info("dry_run_stop", extra={"msg": "Skipping embed + storage in dry-run mode"})
        return

    # ── 5. Embed ──────────────────────────────────────────────────────────
    with stage_timer("embed", logger) as ctx:
        try:
            backend = OpenAIEmbedBackend()
            embed_cfg = EmbedConfig(batch_size=256)
            embedded_chunks = embed_chunks(chunks, backend, embed_cfg)
            metrics.chunks_embedded = len(embedded_chunks)
            ctx["embedded"] = len(embedded_chunks)
        except Exception as exc:
            metrics.embed_errors += 1
            logger.error("embed_failed", extra={"error": str(exc)})
            raise

    # ── 6. Store & Index ──────────────────────────────────────────────────
    with stage_timer("store", logger) as ctx:
        # Ensure indexes exist (idempotent)
        setup_indexes(storage_config)

        result = upsert_chunks(embedded_chunks, storage_config)
        metrics.chunks_upserted = result["upserted"]
        metrics.chunks_matched = result["matched"]
        metrics.storage_errors = result["errors"]
        ctx.update(result)

    # ── 7. Delete stale ───────────────────────────────────────────────────
    if delete_stale:
        with stage_timer("delete_stale", logger) as ctx:
            deleted = hard_delete_stale(stale_days, storage_config)
            metrics.chunks_deleted = deleted
            ctx["deleted"] = deleted


# ---------------------------------------------------------------------------
# Update and delete helpers (for scheduled partial runs)
# ---------------------------------------------------------------------------

def run_update(source: str, new_documents: list[dict], storage_config: StorageConfig) -> dict:
    """
    Re-ingest documents from a single source (incremental update).
    Existing chunks for this source are soft-deleted before re-ingestion.
    """
    logger.info("update_start", extra={"source": source})
    soft_delete_by_source(source, storage_config)

    valid, rejected = clean_batch(new_documents)
    enriched = enrich_batch(valid)
    chunks = chunk_batch(enriched)
    embedded = embed_chunks(chunks)
    result = upsert_chunks(embedded, storage_config)
    logger.info("update_done", extra={"source": source, **result})
    return result


def run_delete(source: str, storage_config: StorageConfig) -> int:
    """Soft-delete all chunks belonging to `source`."""
    deleted = soft_delete_by_source(source, storage_config)
    logger.info("source_deleted", extra={"source": source, "count": deleted})
    return deleted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data Ingestion Batch Job")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkStrategy],
        default="recursive",
        help="Chunking strategy (default: recursive)",
    )
    parser.add_argument("--chunk-size",   type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--dry-run",      action="store_true",
                        help="Run fetch/clean/chunk only; skip embed + storage")
    parser.add_argument("--delete-stale", action="store_true",
                        help="Hard-delete soft-deleted docs older than --stale-days")
    parser.add_argument("--stale-days",   type=int, default=30)
    parser.add_argument("--report-path",  default=None,
                        help="Path to save JSON run report (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_id = str(uuid.uuid4())[:8]
    metrics = PipelineMetrics(run_id=run_id)
    logger.info("pipeline_start", extra={"run_id": run_id})

    storage_config = StorageConfig(
        uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        db_name=os.getenv("DB_NAME", "rag_pipeline"),
        collection_name=os.getenv("COLLECTION_NAME", "chunks"),
    )

    try:
        run_pipeline(
            strategy=ChunkStrategy(args.strategy),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            storage_config=storage_config,
            dry_run=args.dry_run,
            delete_stale=args.delete_stale,
            stale_days=args.stale_days,
            metrics=metrics,
        )
    finally:
        metrics.finish()
        fired = evaluate_alerts(metrics, logger=logger)
        report = RunReport(metrics)
        print(report.summary())
        if args.report_path:
            report.save(args.report_path)
        logger.info("pipeline_done", extra=metrics.to_dict())

    # Exit non-zero if critical alerts fired
    if any(a in fired for a in ("no_chunks_produced", "storage_errors")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
