# 05 – Data Ingestion Pipeline

A production-grade batch ingestion pipeline that cleans, enriches, chunks,
embeds, and stores documents in MongoDB Atlas with full vector search indexing,
similarity metrics, and monitoring.

## Architecture

```
Raw Documents
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Cleaner   │───▶│   Enricher   │───▶│   Chunker   │───▶│   Embedder  │───▶│   Storage   │
│             │    │              │    │             │    │             │    │             │
│ • Unicode   │    │ • word_count │    │ • fixed     │    │ • OpenAI    │    │ • upsert    │
│ • control   │    │ • keywords   │    │ • sentence  │    │   batch API │    │ • soft del  │
│   chars     │    │ • language   │    │ • recursive │    │ • retry +   │    │ • vector    │
│ • boiler-   │    │ • readabil-  │    │ • semantic  │    │   backoff   │    │   index     │
│   plate     │    │   ity tier   │    │             │    │ • cache     │    │ • text      │
│ • hash      │    │ • hash       │    │             │    │             │    │   index     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                                                                                      │
                                                                              ┌───────┴───────┐
                                                                              │  Monitoring   │
                                                                              │               │
                                                                              │ • JSON logs   │
                                                                              │ • stage timer │
                                                                              │ • alert rules │
                                                                              │ • run report  │
                                                                              └───────────────┘
```

## Modules

| Module | File | Responsibility |
|---|---|---|
| Cleaner | `pipeline/cleaner.py` | Unicode normalisation, control-char removal, boilerplate stripping, SHA-256 hash |
| Enricher | `pipeline/enricher.py` | Word/sentence counts, language detection, keyword extraction, readability tier |
| Chunker | `pipeline/chunker.py` | Fixed-size, sentence, recursive, semantic chunking with configurable overlap |
| Embedder | `pipeline/embedder.py` | Batched OpenAI embedding API calls with retry, caching, pluggable backend |
| Storage | `pipeline/storage.py` | Idempotent upsert, soft/hard delete, vector + text + operational index management |
| Similarity | `pipeline/similarity.py` | Cosine, Euclidean, Dot Product, Manhattan, Jaccard; `$vectorSearch` and hybrid RRF pipelines |
| Metrics | `monitoring/metrics.py` | JSON-structured logging, per-stage timers, alert rules, run report |
| Batch Job | `batch_job.py` | Orchestrator CLI wiring all stages together |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# edit .env with your MONGODB_URI and OPENAI_API_KEY

# 3. Dry run (no API calls, no DB writes)
python batch_job.py --dry-run

# 4. Full ingest with recursive chunking (default)
python batch_job.py

# 5. Use sentence-based chunking with custom sizes
python batch_job.py --strategy sentence --chunk-size 300 --chunk-overlap 30

# 6. Ingest + purge stale deleted docs older than 7 days
python batch_job.py --delete-stale --stale-days 7

# 7. Save a JSON run report
python batch_job.py --report-path run_report.json
```

## Chunking Strategies

| Strategy | Best for |
|---|---|
| `recursive` | General text; LangChain-compatible split hierarchy |
| `fixed_size` | Uniform context windows; token-budget-sensitive use cases |
| `sentence` | Narrative text; preserves sentence boundaries |
| `semantic` | High-quality retrieval; groups topically related sentences |

## Similarity Metrics

```python
from pipeline.similarity import compute_similarity, SimilarityMetric

score = compute_similarity(vec_a, vec_b, SimilarityMetric.COSINE)
```

| Metric | Use when |
|---|---|
| `cosine` | General-purpose; most embedding models are trained for this |
| `dot_product` | Unit-norm vectors; faster than cosine |
| `euclidean` | Distance-aware tasks |
| `manhattan` | Sparse vectors |
| `jaccard` | Binary / bag-of-words representations |

## MongoDB Index Setup

The pipeline auto-creates three index types on first run:

1. **Vector Search index** (`vectorSearch`) – ANN search over the `embedding` field
2. **Text Search index** (`search`) – BM25 full-text search for hybrid retrieval
3. **Operational indexes** – unique content_hash+chunk_index, source, status, timestamps

## Monitoring

Every pipeline run emits structured JSON log lines to stdout:

```json
{"ts":"2024-01-15T10:30:00Z","level":"INFO","logger":"batch_job","msg":"stage_end",
 "stage":"embed","duration_s":4.231,"embedded":47}
```

At the end of each run a human-readable summary is printed:

```
============================================================
  Pipeline Run Report  |  run_id=a3f2c1b0
============================================================
  Duration          : 12.4s
  Docs fetched      : 9
  Docs cleaned      : 9
  Docs rejected     : 0 (0.0%)
  Chunks created    : 47
  Chunks embedded   : 47
  Chunks upserted   : 47
  Chunks matched    : 0
  Chunks deleted    : 0
  Embed errors      : 0
  Storage errors    : 0

  Stage timings:
    fetch                 : 0.001s
    clean                 : 0.002s
    enrich                : 0.003s
    chunk                 : 0.004s
    embed                 : 4.231s
    store                 : 1.842s
============================================================
```

Default alert rules fire when:
- Rejection rate > 20 %
- Any embed or storage errors occur
- Zero chunks produced

## Running Tests

```bash
# From the 05-data-ingestion-pipeline directory
python -m pytest tests/ -v
```

All tests run without network access (no MongoDB or OpenAI calls).
