# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-16

Initial release. The tool has been test-driven since day one and ships
with a `test_doc_drift.py` guard that keeps this changelog + README in
sync with the code.

### Added

- **Metrics** (`ragcheck.metrics`): `recall@k`, `precision@k`,
  `hit_rate@k`, `mean_reciprocal_rank`, `dcg_at_k`, `ndcg_at_k`,
  `context_precision`, `context_recall`, `f1_at_k`, `average_precision`,
  `aggregate_metric`. All hand-verified in `tests/test_metrics.py`.
- **Chunkers** (`ragcheck.chunkers`): 5 built-in (`fixed-token`,
  `sliding-window`, `sentence`, `semantic-boundary`, `structural-markdown`)
  plus a pluggable registry (`register_chunker`, `get_chunker`,
  `list_chunkers`).
- **Embedders** (`ragcheck.embedders`): `HashEmbedder` (deterministic,
  offline, default), `SentenceTransformersEmbedder` (lazy-imported),
  `OpenAIEmbedder` (lazy-imported, disk-cached by sha1), `NumpyEmbedder`
  (pre-computed vectors from directory or dict).
- **Diagnostics** (`ragcheck.diagnostics`): `coverage`,
  `duplicate_ratio`, `orphan_chunks`, 7-bucket `size_histogram`, ASCII
  histogram formatter.
- **Runner** (`ragcheck.runner`): `run_evaluation`, `RunConfig`,
  `RunResult`, `dump_result_json`, `load_result_json`. Byte-identical
  output across runs is enforced by
  `tests/test_runner.py::TestDeterminism`.
- **Diff** (`ragcheck.diff`): `diff_runs`, `MetricDelta`, `DiffResult`,
  default per-metric thresholds, `render_diff_markdown`,
  `dump_diff_json`. Exit 1 on any degraded metric with
  `--fail-on-degraded`.
- **Synth** (`ragcheck.synth`): generate a synthetic gold set from an
  unlabeled corpus using `HashEmbedder`-based distinctiveness scoring.
- **Report** (`ragcheck.report`): deterministic JSON renderer,
  Markdown renderer, HTML renderer (with escaping) for run payloads.
- **Bench** (`ragcheck.bench`): `run_bench`, `format_bench_table`,
  `BenchResult`. Completes in under a second on CPU across all three
  bundled fixtures.
- **Fixtures** (`ragcheck.fixtures`): BEIR FiQA subset (8 docs / 10
  queries), MS MARCO subset (6 passages / 8 queries), synthetic
  needle-in-haystack generator.
- **CLI** (`ragcheck.cli`): `run`, `diff`, `bench`, `synth`, `report`,
  `chunkers list`, `embedders list`. Exit codes: `0` OK, `1` regression
  or internal error, `2` usage error.
- **Doc drift guard** (`tests/test_doc_drift.py`): README test-count
  badge + sentence must match the pytest-collected total, every CLI
  subcommand must appear in the README, sibling projects
  (ctxpack / ctxlens / promptdiff / mocklm) must be cited.

### Determinism contract

- All chunk IDs are `sha1(doc_id + start + end + flavor)`.
- All top-k selections use `numpy.argsort(kind="stable")` for deterministic
  tie-breaking.
- All run JSON is emitted with `sort_keys=True`, `ensure_ascii=False`, and
  6-decimal float formatting. No timestamps by default.
- Two runs of `ragcheck run` on the same inputs produce byte-identical
  JSON, verified in test.

### Quality gates at 1.0.0

- 342 tests, 96% branch coverage
- `ruff check ragcheck/` clean
- `mypy ragcheck/` clean
- `ragcheck bench` 0.042s end-to-end on CPU
- Byte-identical determinism verified per run pair
