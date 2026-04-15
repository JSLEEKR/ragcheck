"""Deterministic JSON / HTML / Markdown report renderers."""
from __future__ import annotations

import json
from typing import Any, List, Mapping

from ragcheck.diagnostics import format_histogram_ascii


def render_json(data: Any, *, pretty: bool = True) -> str:
    """Render any JSON-serialisable payload (dict or list) as deterministic JSON text."""
    if pretty:
        return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    return json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"


def render_markdown(run: Mapping[str, Any]) -> str:
    """Render a run result as Markdown.

    Includes the config snapshot, aggregate summary table, chunking
    diagnostics, and a per-query excerpt (first 20 queries).
    """
    lines: List[str] = []
    config = run.get("config", {}) or {}
    summary = run.get("summary", {}) or {}
    diagnostics = run.get("diagnostics", {}) or {}
    corpus_stats = run.get("corpus_stats", {}) or {}
    per_query = run.get("per_query", []) or []
    label = config.get("label") or "run"
    lines.append(f"# ragcheck report: {label}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Corpus | `{config.get('corpus_path', '')}` |")
    lines.append(f"| Gold set | `{config.get('gold_path', '')}` |")
    lines.append(f"| Chunker | `{config.get('chunker', '')}` |")
    lines.append(f"| Embedder | `{config.get('embedder', '')}` |")
    lines.append(f"| k values | `{config.get('k_values', [])}` |")
    lines.append(f"| Seed | `{config.get('seed', '')}` |")
    lines.append("")
    lines.append("## Corpus stats")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k in sorted(corpus_stats.keys()):
        lines.append(f"| {k} | `{corpus_stats[k]}` |")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k in sorted(summary.keys()):
        value = summary[k]
        if value is None:
            lines.append(f"| {k} | null |")
        else:
            lines.append(f"| {k} | {float(value):.6f} |")
    lines.append("")
    if diagnostics:
        lines.append("## Chunking diagnostics")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for k in sorted(diagnostics.keys()):
            if k == "size_histogram":
                continue
            lines.append(f"| {k} | `{diagnostics[k]}` |")
        lines.append("")
        histogram = diagnostics.get("size_histogram") or {}
        if histogram:
            lines.append("### Size histogram")
            lines.append("")
            lines.append("```")
            lines.append(format_histogram_ascii(histogram))
            lines.append("```")
            lines.append("")
    if per_query:
        lines.append("## Per-query metrics (first 20)")
        lines.append("")
        metric_keys: List[str] = []
        if per_query:
            metric_keys = sorted((per_query[0].get("metrics") or {}).keys())
        if metric_keys:
            lines.append("| query_id | " + " | ".join(metric_keys) + " |")
            lines.append("|---|" + "---|" * len(metric_keys))
            for pq in per_query[:20]:
                metrics = pq.get("metrics") or {}
                row = [str(pq.get("query_id", ""))]
                for mk in metric_keys:
                    v = metrics.get(mk)
                    row.append(f"{float(v):.6f}" if v is not None else "null")
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def render_html(run: Mapping[str, Any]) -> str:
    """Render a run result as a self-contained HTML page.

    The rendering is deterministic: no timestamps (unless present in `run`),
    dicts are sorted, and the template includes no random IDs.
    """
    config = run.get("config", {}) or {}
    summary = run.get("summary", {}) or {}
    diagnostics = run.get("diagnostics", {}) or {}
    corpus_stats = run.get("corpus_stats", {}) or {}
    per_query = run.get("per_query", []) or []
    label = config.get("label") or "run"

    def esc(s: Any) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def row(k: str, v: Any) -> str:
        return f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>"

    summary_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{float(summary[k]):.6f}</td></tr>"
        for k in sorted(summary.keys())
        if summary[k] is not None
    )
    config_rows = "".join(row(k, config[k]) for k in sorted(config.keys()))
    corpus_rows = "".join(row(k, corpus_stats[k]) for k in sorted(corpus_stats.keys()))
    diag_rows = "".join(
        row(k, diagnostics[k])
        for k in sorted(diagnostics.keys())
        if k != "size_histogram"
    )
    histogram = diagnostics.get("size_histogram") or {}
    histogram_rows = "".join(
        f"<tr><td>{esc(b)}</td><td>{histogram[b]}</td></tr>"
        for b in sorted(histogram.keys())
    )

    per_query_html = ""
    if per_query:
        metric_keys: List[str] = []
        if per_query:
            metric_keys = sorted((per_query[0].get("metrics") or {}).keys())
        header_cells = "".join(f"<th>{esc(k)}</th>" for k in metric_keys)
        rows_html: List[str] = []
        for pq in per_query:
            metrics = pq.get("metrics") or {}
            cells = "".join(
                f"<td>{float(metrics.get(k, 0.0)):.6f}</td>" for k in metric_keys
            )
            rows_html.append(
                f"<tr><td>{esc(pq.get('query_id', ''))}</td>{cells}</tr>"
            )
        per_query_html = (
            f"<h2>Per-query metrics ({len(per_query)} queries)</h2>"
            f"<table><thead><tr><th>query_id</th>{header_cells}</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ragcheck report: {esc(label)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1, h2 {{ color: #2a4365; }}
table {{ border-collapse: collapse; margin-bottom: 2em; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: left; font-size: 14px; }}
th {{ background: #f8f9fb; }}
code, td, th {{ font-family: 'SF Mono', Consolas, monospace; }}
.muted {{ color: #666; }}
</style>
</head>
<body>
<h1>ragcheck report: {esc(label)}</h1>
<p class="muted">tool_version={esc(run.get("tool_version", ""))}, schema_version={esc(run.get("schema_version", 1))}</p>

<h2>Configuration</h2>
<table><tbody>{config_rows}</tbody></table>

<h2>Corpus stats</h2>
<table><tbody>{corpus_rows}</tbody></table>

<h2>Aggregate metrics</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{summary_rows}</tbody></table>

<h2>Chunking diagnostics</h2>
<table><tbody>{diag_rows}</tbody></table>

<h3>Size histogram</h3>
<table><thead><tr><th>Bucket</th><th>Count</th></tr></thead>
<tbody>{histogram_rows}</tbody></table>

{per_query_html}
</body>
</html>
"""
    return html
