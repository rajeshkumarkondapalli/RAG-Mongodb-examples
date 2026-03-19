"""
Unit tests for the data ingestion pipeline (no external dependencies).

Run with:  python -m pytest tests/ -v
"""

import math
import pytest

from pipeline.cleaner import CleaningConfig, clean_document, clean_batch
from pipeline.enricher import enrich_document
from pipeline.chunker import ChunkConfig, ChunkStrategy, chunk_document
from pipeline.similarity import (
    cosine_similarity, euclidean_similarity, dot_product_similarity,
    manhattan_similarity, jaccard_similarity, compute_similarity,
    SimilarityMetric, rank_by_similarity,
)
from monitoring.metrics import PipelineMetrics, evaluate_alerts, AlertRule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RAW = {
    "title": "Test Document",
    "source": "test",
    "category": "unit_test",
    "content": (
        "MongoDB Atlas Vector Search is a powerful feature that allows applications to "
        "perform semantic similarity searches. It integrates natively with embedding models "
        "and supports multiple distance metrics. Chunking strategies improve retrieval quality. "
        "Metadata enrichment adds useful context for filtering and ranking search results."
    ),
}


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------

class TestCleaner:
    def test_basic_clean(self):
        cleaned = clean_document(SAMPLE_RAW)
        assert cleaned.is_valid
        assert len(cleaned.content) > 0
        assert cleaned.content_hash != ""

    def test_control_chars_removed(self):
        raw = {**SAMPLE_RAW, "content": "Hello\x00\x01World\x07!"}
        cleaned = clean_document(raw)
        assert "\x00" not in cleaned.content
        assert "Hello" in cleaned.content

    def test_too_short_rejected(self):
        raw = {**SAMPLE_RAW, "content": "Hi"}
        cleaned = clean_document(raw, CleaningConfig(min_content_length=20))
        assert not cleaned.is_valid
        assert cleaned.rejection_reason is not None

    def test_batch_split(self):
        docs = [SAMPLE_RAW, {**SAMPLE_RAW, "content": "x"}]
        valid, rejected = clean_batch(docs)
        assert len(valid) == 1
        assert len(rejected) == 1

    def test_boilerplate_removed(self):
        raw = {**SAMPLE_RAW, "content": "Some content. © 2024 Acme Corp. More content here."}
        cleaned = clean_document(raw)
        assert "© 2024" not in cleaned.content

    def test_content_hash_stable(self):
        c1 = clean_document(SAMPLE_RAW)
        c2 = clean_document(SAMPLE_RAW)
        assert c1.content_hash == c2.content_hash


# ---------------------------------------------------------------------------
# Enricher tests
# ---------------------------------------------------------------------------

class TestEnricher:
    def test_enrichment_fields(self):
        cleaned = clean_document(SAMPLE_RAW)
        enriched = enrich_document(cleaned)
        meta = enriched.metadata
        assert "word_count" in meta
        assert "char_count" in meta
        assert "keywords" in meta
        assert "language" in meta
        assert "readability" in meta
        assert "ingested_at" in meta
        assert "pipeline_version" in meta

    def test_word_count_positive(self):
        cleaned = clean_document(SAMPLE_RAW)
        enriched = enrich_document(cleaned)
        assert enriched.metadata["word_count"] > 0

    def test_keywords_are_list(self):
        cleaned = clean_document(SAMPLE_RAW)
        enriched = enrich_document(cleaned)
        assert isinstance(enriched.metadata["keywords"], list)

    def test_language_english(self):
        cleaned = clean_document(SAMPLE_RAW)
        enriched = enrich_document(cleaned)
        assert enriched.metadata["language"] == "en"

    def test_extra_metadata_merged(self):
        cleaned = clean_document(SAMPLE_RAW)
        enriched = enrich_document(cleaned, extra_metadata={"author": "Alice"})
        assert enriched.metadata["author"] == "Alice"


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    def _enriched(self):
        return enrich_document(clean_document(SAMPLE_RAW))

    def test_recursive_produces_chunks(self):
        doc = self._enriched()
        cfg = ChunkConfig(strategy=ChunkStrategy.RECURSIVE, chunk_size=200, chunk_overlap=20)
        chunks = chunk_document(doc, cfg)
        assert len(chunks) > 0

    def test_fixed_size_chunks(self):
        doc = self._enriched()
        cfg = ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=100, chunk_overlap=10)
        chunks = chunk_document(doc, cfg)
        # All chunks except possibly the last should be <= 100+10 chars
        for c in chunks[:-1]:
            assert len(c.content) <= 120

    def test_sentence_chunks(self):
        doc = self._enriched()
        cfg = ChunkConfig(strategy=ChunkStrategy.SENTENCE, chunk_size=300)
        chunks = chunk_document(doc, cfg)
        assert len(chunks) > 0

    def test_chunk_metadata_populated(self):
        doc = self._enriched()
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.metadata["chunk_strategy"] is not None
            assert c.metadata["chunk_total"] == len(chunks)

    def test_chunk_indices_sequential(self):
        doc = self._enriched()
        chunks = chunk_document(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_min_chunk_length_filter(self):
        doc = self._enriched()
        cfg = ChunkConfig(min_chunk_length=1000)   # very large → no chunks pass
        chunks = chunk_document(doc, cfg)
        assert len(chunks) == 0

    def test_semantic_requires_embed_fn(self):
        doc = self._enriched()
        cfg = ChunkConfig(strategy=ChunkStrategy.SEMANTIC)
        with pytest.raises(ValueError):
            chunk_document(doc, cfg)


# ---------------------------------------------------------------------------
# Similarity tests
# ---------------------------------------------------------------------------

class TestSimilarity:
    A = [1.0, 0.0, 0.0]
    B = [0.0, 1.0, 0.0]
    C = [1.0, 0.0, 0.0]  # identical to A

    def test_cosine_identical(self):
        assert cosine_similarity(self.A, self.C) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert cosine_similarity(self.A, self.B) == pytest.approx(0.0)

    def test_euclidean_identical(self):
        assert euclidean_similarity(self.A, self.C) == pytest.approx(1.0)

    def test_euclidean_different(self):
        sim = euclidean_similarity(self.A, self.B)
        assert 0.0 < sim < 1.0

    def test_dot_product(self):
        assert dot_product_similarity(self.A, self.C) == pytest.approx(1.0)
        assert dot_product_similarity(self.A, self.B) == pytest.approx(0.0)

    def test_manhattan_identical(self):
        assert manhattan_similarity(self.A, self.C) == pytest.approx(1.0)

    def test_jaccard(self):
        a = [0.9, 0.1, 0.0]
        b = [0.8, 0.0, 0.0]
        j = jaccard_similarity(a, b, threshold=0.5)
        assert j == pytest.approx(1.0)   # both only have dim-0 above threshold

    def test_compute_similarity_dispatch(self):
        for metric in SimilarityMetric:
            score = compute_similarity(self.A, self.C, metric)
            assert isinstance(score, float)

    def test_rank_by_similarity(self):
        candidates = [
            {"id": 1, "embedding": [1.0, 0.0, 0.0]},
            {"id": 2, "embedding": [0.0, 1.0, 0.0]},
            {"id": 3, "embedding": [0.9, 0.1, 0.0]},
        ]
        query = [1.0, 0.0, 0.0]
        ranked = rank_by_similarity(query, candidates, top_k=2)
        assert len(ranked) == 2
        assert ranked[0]["id"] == 1   # perfect match first


# ---------------------------------------------------------------------------
# Metrics / monitoring tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_rejection_rate(self):
        m = PipelineMetrics(run_id="test")
        m.docs_fetched = 10
        m.docs_rejected = 2
        assert m.rejection_rate == pytest.approx(0.2)

    def test_alert_fires_on_high_rejection(self):
        m = PipelineMetrics(run_id="test")
        m.docs_fetched = 10
        m.docs_rejected = 5   # 50% > 20% threshold
        rules = [AlertRule("high_rejection_rate", "rejection_rate", threshold=0.20)]
        fired = evaluate_alerts(m, rules)
        assert "high_rejection_rate" in fired

    def test_alert_does_not_fire_on_low_rejection(self):
        m = PipelineMetrics(run_id="test")
        m.docs_fetched = 100
        m.docs_rejected = 1
        rules = [AlertRule("high_rejection_rate", "rejection_rate", threshold=0.20)]
        fired = evaluate_alerts(m, rules)
        assert fired == []

    def test_metrics_to_dict(self):
        m = PipelineMetrics(run_id="xyz")
        m.finish()
        d = m.to_dict()
        assert d["run_id"] == "xyz"
        assert "duration_s" in d
