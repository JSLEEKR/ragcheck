"""Command-line interface for ragcheck.

Commands:
- `ragcheck run --corpus DIR --gold FILE --out report.json`
- `ragcheck diff baseline.json head.json --out diff.json`
- `ragcheck bench`
- `ragcheck synth --corpus DIR --questions N --out gold.json`
- `ragcheck report --in run.json --format html|md|json`
- `ragcheck chunkers list`
- `ragcheck embedders list`

Exit codes:
- 0: success (including a clean diff with no degraded metrics)
- 1: regression detected (diff) OR runtime failure
- 2: usage error (bad flags, missing file, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ragcheck import __version__
from ragcheck.bench import format_bench_table, run_bench
from ragcheck.chunkers import list_chunkers
from ragcheck.corpus import save_gold_set
from ragcheck.diff import (
    diff_runs,
    dump_diff_json,
    render_diff_markdown,
)
from ragcheck.embedders import list_embedders
from ragcheck.report import render_html, render_json, render_markdown
from ragcheck.runner import RunConfig, dump_result_json, load_result_json, run_evaluation
from ragcheck.synth import synth_gold_set

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragcheck",
        description="Offline RAG retrieval quality harness (recall@k, nDCG, MRR, diffs).",
    )
    p.add_argument("--version", action="version", version=f"ragcheck {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = False  # we handle it manually so `ragcheck` alone prints help

    # run
    pr = sub.add_parser("run", help="Compute retrieval metrics on a labeled gold set")
    pr.add_argument("--corpus", required=True, help="Directory of .txt / .md files")
    pr.add_argument("--gold", required=True, help="Gold set JSON path")
    pr.add_argument("--out", required=True, help="Output JSON path")
    pr.add_argument("--chunker", default="fixed-token")
    pr.add_argument("--chunker-args", default="{}", help="JSON dict of chunker kwargs")
    pr.add_argument("--embedder", default="hash")
    pr.add_argument("--embedder-args", default="{}", help="JSON dict of embedder kwargs")
    pr.add_argument("--k", default="1,3,5,10", help="comma-separated k values")
    pr.add_argument("--label", default="")
    pr.add_argument("--include-timestamp", action="store_true")
    pr.add_argument("--no-per-query", action="store_true")
    pr.add_argument("--no-diagnostics", action="store_true")
    pr.add_argument("--compact", action="store_true", help="Minified JSON output")

    # diff
    pd = sub.add_parser(
        "diff",
        help="Regression diff between two run JSON files (exit 1 if degraded)",
    )
    pd.add_argument("baseline", help="Baseline run JSON path")
    pd.add_argument("head", help="Head (candidate) run JSON path")
    pd.add_argument("--out", default="", help="Write diff JSON here (otherwise stdout)")
    pd.add_argument("--markdown", default="", help="Also render a Markdown diff here")
    pd.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="metric_prefix=value (repeatable), e.g. --threshold recall=0.03",
    )
    pd.add_argument(
        "--no-fail",
        action="store_true",
        help="Never return non-zero exit code, even on degradation",
    )

    # bench
    pb = sub.add_parser("bench", help="Run bundled fixtures (30-second smoke test)")
    pb.add_argument("--json", action="store_true", help="Emit bench results as JSON")

    # synth
    ps = sub.add_parser("synth", help="Bootstrap a synthetic gold set from a corpus")
    ps.add_argument("--corpus", required=True, help="Directory of .txt / .md files")
    ps.add_argument("--questions", type=int, default=50)
    ps.add_argument("--out", required=True)
    ps.add_argument("--seed", type=int, default=42)

    # report
    prp = sub.add_parser("report", help="Render a run JSON as HTML / Markdown / JSON")
    prp.add_argument("--in", dest="inp", required=True, help="Input run JSON path")
    prp.add_argument("--format", choices=["html", "md", "json"], default="md")
    prp.add_argument("--out", default="", help="Output path (stdout if empty)")

    # chunkers
    pc = sub.add_parser("chunkers", help="List or describe chunkers")
    pc.add_argument("action", choices=["list"], default="list", nargs="?")

    # embedders
    pe = sub.add_parser("embedders", help="List or describe embedders")
    pe.add_argument("action", choices=["list"], default="list", nargs="?")

    return p


def _parse_kv_thresholds(raw: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"bad --threshold value {entry!r}; expected metric_prefix=float"
            )
        key, val = entry.split("=", 1)
        try:
            out[key.strip()] = float(val.strip())
        except ValueError as e:
            raise ValueError(f"bad --threshold value {entry!r}: {e}") from e
    return out


def _parse_k_list(raw: str) -> List[int]:
    ks: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError as e:
            raise ValueError(f"bad --k value {part!r}; must be integer") from e
        if v <= 0:
            raise ValueError(f"bad --k value {v}; must be > 0")
        ks.append(v)
    if not ks:
        raise ValueError("--k requires at least one positive integer")
    return sorted(set(ks))


def _parse_json_args(raw: str, flag_name: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"bad {flag_name} JSON: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object")
    return value


def _write_out(text: str, out_path: str, stdout) -> None:
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    else:
        stdout.write(text)
        if not text.endswith("\n"):
            stdout.write("\n")


def _cmd_run(args: argparse.Namespace, stdout, stderr) -> int:
    try:
        ks = _parse_k_list(args.k)
        chunker_args = _parse_json_args(args.chunker_args, "--chunker-args")
        embedder_args = _parse_json_args(args.embedder_args, "--embedder-args")
    except ValueError as e:
        stderr.write(f"ragcheck: {e}\n")
        return EXIT_USAGE
    corpus_path = Path(args.corpus)
    gold_path = Path(args.gold)
    if not corpus_path.exists():
        stderr.write(f"ragcheck: corpus not found: {corpus_path}\n")
        return EXIT_USAGE
    if not gold_path.exists():
        stderr.write(f"ragcheck: gold set not found: {gold_path}\n")
        return EXIT_USAGE
    out_path = Path(args.out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        stderr.write(f"ragcheck: cannot create output directory {out_path.parent}: {e}\n")
        return EXIT_USAGE
    config = RunConfig(
        corpus_path=str(corpus_path),
        gold_path=str(gold_path),
        chunker=args.chunker,
        chunker_args=chunker_args,
        embedder=args.embedder,
        embedder_args=embedder_args,
        k_values=ks,
        label=args.label,
        include_timestamp=args.include_timestamp,
        include_per_query=not args.no_per_query,
        include_diagnostics=not args.no_diagnostics,
    )
    try:
        result = run_evaluation(config=config)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        stderr.write(f"ragcheck: {e}\n")
        return EXIT_USAGE
    try:
        dump_result_json(result, out_path, pretty=not args.compact)
    except (OSError, PermissionError) as e:
        stderr.write(f"ragcheck: cannot write output {out_path}: {e}\n")
        return EXIT_USAGE
    stdout.write(f"ragcheck: wrote {out_path}\n")
    # Brief summary
    for k in sorted(result.summary.keys()):
        stdout.write(f"  {k}: {result.summary[k]:.6f}\n")
    return EXIT_OK


def _cmd_diff(args: argparse.Namespace, stdout, stderr) -> int:
    try:
        thresholds = _parse_kv_thresholds(args.threshold)
    except ValueError as e:
        stderr.write(f"ragcheck: {e}\n")
        return EXIT_USAGE
    baseline_path = Path(args.baseline)
    head_path = Path(args.head)
    if not baseline_path.exists():
        stderr.write(f"ragcheck: baseline not found: {baseline_path}\n")
        return EXIT_USAGE
    if not head_path.exists():
        stderr.write(f"ragcheck: head not found: {head_path}\n")
        return EXIT_USAGE
    try:
        baseline = load_result_json(baseline_path)
        head = load_result_json(head_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        stderr.write(f"ragcheck: cannot load run JSON: {e}\n")
        return EXIT_USAGE
    diff = diff_runs(baseline, head, thresholds=thresholds, fail_on_degraded=True)
    if args.out:
        try:
            dump_diff_json(diff, Path(args.out))
        except (OSError, PermissionError) as e:
            stderr.write(f"ragcheck: cannot write {args.out}: {e}\n")
            return EXIT_USAGE
        stdout.write(f"ragcheck: wrote {args.out}\n")
    if args.markdown:
        md_path = Path(args.markdown)
        try:
            md_path.parent.mkdir(parents=True, exist_ok=True)
            with md_path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(render_diff_markdown(diff))
        except (OSError, PermissionError) as e:
            stderr.write(f"ragcheck: cannot write {md_path}: {e}\n")
            return EXIT_USAGE
        stdout.write(f"ragcheck: wrote {md_path}\n")
    if not args.out and not args.markdown:
        # Default: print Markdown summary to stdout (render once)
        rendered = render_diff_markdown(diff)
        stdout.write(rendered)
        if not rendered.endswith("\n"):
            stdout.write("\n")
    if args.no_fail:
        return EXIT_OK
    return diff.exit_code


def _cmd_bench(args: argparse.Namespace, stdout, stderr) -> int:
    try:
        results = run_bench()
    except Exception as e:  # pragma: no cover - defensive
        stderr.write(f"ragcheck: bench failed: {e}\n")
        return EXIT_FAIL
    if args.json:
        payload = [r.to_dict() for r in results]
        stdout.write(render_json(payload))
    else:
        stdout.write(format_bench_table(results))
        stdout.write("\n")
    return EXIT_OK


def _cmd_synth(args: argparse.Namespace, stdout, stderr) -> int:
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        stderr.write(f"ragcheck: corpus not found: {corpus_path}\n")
        return EXIT_USAGE
    if args.questions <= 0:
        stderr.write("ragcheck: --questions must be > 0\n")
        return EXIT_USAGE
    try:
        gold = synth_gold_set(corpus_path, n_questions=args.questions, seed=args.seed)
    except (FileNotFoundError, ValueError) as e:
        stderr.write(f"ragcheck: {e}\n")
        return EXIT_USAGE
    out_path = Path(args.out)
    try:
        save_gold_set(gold, out_path)
    except (OSError, PermissionError) as e:
        stderr.write(f"ragcheck: cannot write {out_path}: {e}\n")
        return EXIT_USAGE
    stdout.write(
        f"ragcheck: wrote {out_path} with {len(gold.questions)} synthetic questions\n"
    )
    stdout.write(
        "ragcheck: NOTE — these questions are synthetic, not human-graded. "
        "Use for smoke-testing only.\n"
    )
    return EXIT_OK


def _cmd_report(args: argparse.Namespace, stdout, stderr) -> int:
    in_path = Path(args.inp)
    if not in_path.exists():
        stderr.write(f"ragcheck: input not found: {in_path}\n")
        return EXIT_USAGE
    try:
        run = load_result_json(in_path)
    except (OSError, json.JSONDecodeError) as e:
        stderr.write(f"ragcheck: cannot load {in_path}: {e}\n")
        return EXIT_USAGE
    if args.format == "html":
        text = render_html(run)
    elif args.format == "md":
        text = render_markdown(run)
    else:
        text = render_json(run)
    try:
        _write_out(text, args.out, stdout)
    except (OSError, PermissionError) as e:
        stderr.write(f"ragcheck: cannot write output: {e}\n")
        return EXIT_USAGE
    return EXIT_OK


def _cmd_chunkers(args: argparse.Namespace, stdout, stderr) -> int:
    for name in list_chunkers():
        stdout.write(name + "\n")
    return EXIT_OK


def _cmd_embedders(args: argparse.Namespace, stdout, stderr) -> int:
    for name in list_embedders():
        stdout.write(name + "\n")
    return EXIT_OK


COMMANDS = {
    "run": _cmd_run,
    "diff": _cmd_diff,
    "bench": _cmd_bench,
    "synth": _cmd_synth,
    "report": _cmd_report,
    "chunkers": _cmd_chunkers,
    "embedders": _cmd_embedders,
}


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout=None,
    stderr=None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    if not args.command:
        parser.print_help(stdout)
        return EXIT_USAGE
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help(stderr)
        return EXIT_USAGE
    return handler(args, stdout, stderr)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
