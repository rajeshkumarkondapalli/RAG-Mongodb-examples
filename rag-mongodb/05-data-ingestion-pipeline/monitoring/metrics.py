"""
Monitoring & Observability Module
----------------------------------
Lightweight, zero-dependency metrics and structured logging for the pipeline.

Components
----------
PipelineMetrics  – dataclass that accumulates counters and timings per run
StageTimer       – context manager for per-stage wall-clock timing
MetricsLogger    – emits structured JSON log lines (stdout or file)
RunReport        – final human-readable summary + JSON export
AlertRule        – simple threshold-based alert evaluation
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                log_obj[key] = val
        return json.dumps(log_obj, default=str)


def get_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Stage timing context manager
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str
    duration_s: float
    success: bool
    error: str | None = None


@contextmanager
def stage_timer(stage_name: str, logger: logging.Logger | None = None) -> Generator[dict, None, None]:
    """
    Context manager that times a pipeline stage and logs start/end.

    Usage::

        with stage_timer("cleaning", logger) as ctx:
            ctx["docs_processed"] = 42
    """
    ctx: dict[str, Any] = {}
    start = time.perf_counter()
    logger = logger or get_logger()
    logger.info("stage_start", extra={"stage": stage_name})
    try:
        yield ctx
        duration = time.perf_counter() - start
        logger.info(
            "stage_end",
            extra={"stage": stage_name, "duration_s": round(duration, 3), **ctx},
        )
        ctx["_result"] = StageResult(stage_name, duration, True)
    except Exception as exc:
        duration = time.perf_counter() - start
        logger.error(
            "stage_error",
            extra={"stage": stage_name, "duration_s": round(duration, 3), "error": str(exc)},
            exc_info=True,
        )
        ctx["_result"] = StageResult(stage_name, duration, False, str(exc))
        raise


# ---------------------------------------------------------------------------
# Pipeline metrics accumulator
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    run_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    # Ingestion counts
    docs_fetched: int = 0
    docs_cleaned: int = 0
    docs_rejected: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_upserted: int = 0
    chunks_matched: int = 0   # no-change upserts
    chunks_deleted: int = 0
    embed_errors: int = 0
    storage_errors: int = 0

    # Timing (seconds per stage)
    stage_timings: dict[str, float] = field(default_factory=dict)

    # Alert flags
    alerts: list[str] = field(default_factory=list)

    def record_stage(self, result: StageResult) -> None:
        self.stage_timings[result.stage] = result.duration_s

    @property
    def duration_s(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def rejection_rate(self) -> float:
        total = self.docs_fetched or 1
        return self.docs_rejected / total

    @property
    def embed_error_rate(self) -> float:
        total = self.chunks_created or 1
        return self.embed_errors / total

    def finish(self) -> None:
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        d["rejection_rate"] = self.rejection_rate
        d["embed_error_rate"] = self.embed_error_rate
        return d


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------

@dataclass
class AlertRule:
    name: str
    metric_attr: str          # attribute path on PipelineMetrics
    threshold: float
    operator: str = ">"       # >, <, >=, <=, ==

    def evaluate(self, metrics: PipelineMetrics) -> bool:
        value = getattr(metrics, self.metric_attr, None)
        if value is None:
            return False
        ops = {">": value > self.threshold, "<": value < self.threshold,
               ">=": value >= self.threshold, "<=": value <= self.threshold,
               "==": value == self.threshold}
        return ops.get(self.operator, False)


DEFAULT_ALERT_RULES: list[AlertRule] = [
    AlertRule("high_rejection_rate",  "rejection_rate",   threshold=0.20),
    AlertRule("embed_errors",         "embed_errors",     threshold=0,    operator=">"),
    AlertRule("storage_errors",       "storage_errors",   threshold=0,    operator=">"),
    AlertRule("no_chunks_produced",   "chunks_created",   threshold=1,    operator="<"),
]


def evaluate_alerts(
    metrics: PipelineMetrics,
    rules: list[AlertRule] | None = None,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Evaluate alert rules against metrics; append fired alert names."""
    rules = rules or DEFAULT_ALERT_RULES
    logger = logger or get_logger()
    fired: list[str] = []
    for rule in rules:
        if rule.evaluate(metrics):
            msg = (
                f"ALERT [{rule.name}]: {rule.metric_attr}="
                f"{getattr(metrics, rule.metric_attr, 'N/A')} "
                f"{rule.operator} {rule.threshold}"
            )
            logger.warning(msg, extra={"alert": rule.name})
            fired.append(rule.name)
    metrics.alerts.extend(fired)
    return fired


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------

class RunReport:
    """Pretty-print and JSON-export a completed pipeline run."""

    def __init__(self, metrics: PipelineMetrics) -> None:
        self.metrics = metrics

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"{'='*60}",
            f"  Pipeline Run Report  |  run_id={m.run_id}",
            f"{'='*60}",
            f"  Duration          : {m.duration_s:.1f}s",
            f"  Docs fetched      : {m.docs_fetched}",
            f"  Docs cleaned      : {m.docs_cleaned}",
            f"  Docs rejected     : {m.docs_rejected} ({m.rejection_rate:.1%})",
            f"  Chunks created    : {m.chunks_created}",
            f"  Chunks embedded   : {m.chunks_embedded}",
            f"  Chunks upserted   : {m.chunks_upserted}",
            f"  Chunks matched    : {m.chunks_matched}",
            f"  Chunks deleted    : {m.chunks_deleted}",
            f"  Embed errors      : {m.embed_errors}",
            f"  Storage errors    : {m.storage_errors}",
            f"",
            f"  Stage timings:",
        ]
        for stage, dur in m.stage_timings.items():
            lines.append(f"    {stage:<22}: {dur:.3f}s")
        if m.alerts:
            lines.append(f"")
            lines.append(f"  *** ALERTS FIRED: {', '.join(m.alerts)} ***")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.metrics.to_dict(), indent=2, default=str)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"[report] Saved run report to {path}")
