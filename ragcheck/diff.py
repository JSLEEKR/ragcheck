"""Regression diff: compare two RunResult JSONs and flag significant drops.

Output is deterministic JSON + pretty Markdown. Exit code 1 if any metric
drops past the configured threshold; exit 0 if all flat-or-improved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Default thresholds (absolute, in metric units).
DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Recall/precision/hit_rate: 0.02 absolute = 2 percentage points
    "recall": 0.02,
    "precision": 0.02,
    "hit_rate": 0.02,
    "mrr": 0.02,
    "ndcg": 0.02,
    "context_precision": 0.02,
    "context_recall": 0.02,
}


def _threshold_for(metric: str, thresholds: Mapping[str, float]) -> float:
    """Match a metric name to the most specific threshold key.

    recall@5 -> first tries "recall@5", then "recall", then falls back to
    the smallest numeric default (0.02).
    """
    if metric in thresholds:
        return float(thresholds[metric])
    for key in sorted(thresholds.keys(), key=lambda k: -len(k)):
        if metric.startswith(key):
            return float(thresholds[key])
    return 0.02


@dataclass
class MetricDelta:
    metric: str
    baseline: float
    head: float
    abs_delta: float
    rel_delta: float
    threshold: float
    status: str  # "improved" | "flat" | "degraded" | "new" | "removed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": round(self.baseline, 6),
            "head": round(self.head, 6),
            "abs_delta": round(self.abs_delta, 6),
            "rel_delta": round(self.rel_delta, 6),
            "threshold": round(self.threshold, 6),
            "status": self.status,
        }


@dataclass
class DiffResult:
    baseline_label: str = ""
    head_label: str = ""
    deltas: List[MetricDelta] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    improved: List[str] = field(default_factory=list)
    flat: List[str] = field(default_factory=list)
    exit_code: int = 0
    corpus_changed: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_label": self.baseline_label,
            "head_label": self.head_label,
            "corpus_changed": self.corpus_changed,
            "exit_code": self.exit_code,
            "summary": {
                "degraded": sorted(self.degraded),
                "improved": sorted(self.improved),
                "flat": sorted(self.flat),
            },
            "warnings": sorted(self.warnings),
            "deltas": [d.to_dict() for d in sorted(self.deltas, key=lambda x: x.metric)],
        }


def _rel_delta(baseline: float, head: float) -> float:
    if baseline == 0.0:
        if head == 0.0:
            return 0.0
        # Undefined relative delta; use inf-ish signal as a capped value.
        return 1.0 if head > 0 else -1.0
    return (head - baseline) / abs(baseline)


def diff_runs(
    baseline: Mapping[str, Any],
    head: Mapping[str, Any],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    fail_on_degraded: bool = True,
) -> DiffResult:
    """Compute a metric-wise regression diff between two run payloads.

    Both arguments are dict representations of RunResult (as produced by
    `RunResult.to_dict()`). If metrics present in only one side, they are
    reported with status "new" or "removed".
    """
    th = {**DEFAULT_THRESHOLDS, **(dict(thresholds) if thresholds else {})}
    base_summary = dict(baseline.get("summary", {}))
    head_summary = dict(head.get("summary", {}))
    all_metrics = sorted(set(base_summary.keys()) | set(head_summary.keys()))

    result = DiffResult(
        baseline_label=str(baseline.get("config", {}).get("label", "baseline")),
        head_label=str(head.get("config", {}).get("label", "head")),
    )

    # Corpus change detection
    base_sha = str(baseline.get("corpus_stats", {}).get("corpus_sha1", ""))
    head_sha = str(head.get("corpus_stats", {}).get("corpus_sha1", ""))
    if base_sha and head_sha and base_sha != head_sha:
        result.corpus_changed = True

    # Configuration drift warnings. Comparing two runs that measured
    # recall@10 against different ``top_k_cap`` values is meaningless:
    # the run with ``top_k_cap=5`` literally could not report any
    # retrievals in positions 6-10, so recall@10 is artificially
    # truncated. Warn instead of silently accepting the comparison
    # (Cycle E M2). Same logic for mismatched ``k_values`` lists and
    # different chunker or embedder names (which also make metric
    # deltas potentially meaningless but are legitimate comparisons).
    base_cfg = baseline.get("config", {}) or {}
    head_cfg = head.get("config", {}) or {}
    base_cap = base_cfg.get("top_k_cap")
    head_cap = head_cfg.get("top_k_cap")
    if base_cap is not None and head_cap is not None and base_cap != head_cap:
        result.warnings.append(
            f"top_k_cap differs: baseline={base_cap}, head={head_cap}; "
            f"metrics computed over different retrieval windows may not be "
            f"directly comparable"
        )
    base_k = base_cfg.get("k_values")
    head_k = head_cfg.get("k_values")
    if base_k is not None and head_k is not None and base_k != head_k:
        result.warnings.append(
            f"k_values differ: baseline={base_k}, head={head_k}"
        )

    for m in all_metrics:
        if m not in base_summary:
            delta = MetricDelta(
                metric=m,
                baseline=0.0,
                head=float(head_summary[m] or 0.0),
                abs_delta=float(head_summary[m] or 0.0),
                rel_delta=0.0,
                threshold=_threshold_for(m, th),
                status="new",
            )
            result.deltas.append(delta)
            result.improved.append(m)
            continue
        if m not in head_summary:
            delta = MetricDelta(
                metric=m,
                baseline=float(base_summary[m] or 0.0),
                head=0.0,
                abs_delta=-float(base_summary[m] or 0.0),
                rel_delta=-1.0,
                threshold=_threshold_for(m, th),
                status="removed",
            )
            result.deltas.append(delta)
            result.degraded.append(m)
            continue
        b = float(base_summary[m] or 0.0)
        h = float(head_summary[m] or 0.0)
        abs_d = h - b
        rel_d = _rel_delta(b, h)
        thr = _threshold_for(m, th)
        # Round the delta to 6 decimals before comparing against the
        # threshold so exact boundary cases (e.g. 0.80 - 0.78) don't fall on
        # the "degraded" side purely because of IEEE float imprecision. The
        # serialised abs_delta is already rounded to 6 decimals, so this also
        # keeps the classification and the rendered number in sync.
        abs_d_cmp = round(abs_d, 6)
        if abs_d_cmp < -thr:
            status = "degraded"
            result.degraded.append(m)
        elif abs_d_cmp > thr:
            status = "improved"
            result.improved.append(m)
        else:
            status = "flat"
            result.flat.append(m)
        result.deltas.append(
            MetricDelta(
                metric=m,
                baseline=b,
                head=h,
                abs_delta=abs_d,
                rel_delta=rel_d,
                threshold=thr,
                status=status,
            )
        )

    if fail_on_degraded and result.degraded:
        result.exit_code = 1
    else:
        result.exit_code = 0
    return result


def dump_diff_json(diff: DiffResult, path: Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = diff.to_dict()
    with p.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")


def render_diff_markdown(diff: DiffResult) -> str:
    """Render a diff as Markdown for PR comments."""
    lines: List[str] = []
    lines.append("# ragcheck regression diff")
    lines.append("")
    lines.append(f"- **Baseline**: `{diff.baseline_label}`")
    lines.append(f"- **Head**: `{diff.head_label}`")
    if diff.corpus_changed:
        lines.append("- **Corpus changed**: yes (corpus_sha1 differs between runs)")
    lines.append(f"- **Degraded**: {len(diff.degraded)}")
    lines.append(f"- **Improved**: {len(diff.improved)}")
    lines.append(f"- **Flat**: {len(diff.flat)}")
    lines.append("")
    if diff.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in sorted(diff.warnings):
            lines.append(f"- {w}")
        lines.append("")
    if diff.degraded:
        lines.append("## Degraded metrics")
        lines.append("")
        lines.append("| Metric | Baseline | Head | Abs delta | Threshold |")
        lines.append("|---|---|---|---|---|")
        for d in sorted(diff.deltas, key=lambda x: x.metric):
            if d.status != "degraded":
                continue
            lines.append(
                f"| {d.metric} | {d.baseline:.6f} | {d.head:.6f} | "
                f"{d.abs_delta:+.6f} | {d.threshold:.4f} |"
            )
        lines.append("")
    lines.append("## All metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Head | Abs delta | Rel delta | Status |")
    lines.append("|---|---|---|---|---|---|")
    for d in sorted(diff.deltas, key=lambda x: x.metric):
        lines.append(
            f"| {d.metric} | {d.baseline:.6f} | {d.head:.6f} | "
            f"{d.abs_delta:+.6f} | {d.rel_delta:+.4f} | {d.status} |"
        )
    lines.append("")
    if diff.exit_code != 0:
        lines.append(f"**Exit code**: {diff.exit_code} (at least one metric degraded)")
    else:
        lines.append(f"**Exit code**: {diff.exit_code} (all metrics flat-or-improved)")
    lines.append("")
    return "\n".join(lines)
