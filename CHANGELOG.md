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

- 359 tests, 96% branch coverage
- `ruff check ragcheck/` clean
- `mypy ragcheck/` clean
- `ragcheck bench` 0.042s end-to-end on CPU
- Byte-identical determinism verified per run pair

## [1.0.1] - 2026-04-16

### Fixed (Phase 3 Eval Cycle A — 8 bugs)

- **Public API:** `ragcheck.load_result_json` and `ragcheck.dump_result_json`
  are now exported from the top-level package. The README's programmatic
  usage example referenced `ragcheck.load_result_json` which was not
  actually importable at 1.0.0.
- **`render_html`:** per-query metrics with a `None` value no longer crash
  with `TypeError: float() argument must be ... not 'NoneType'`. They now
  render as `null`, matching the Markdown path.
- **`register_chunker`:** duplicate registrations now raise `ValueError`
  instead of silently overwriting the existing factory. The README
  documents this behaviour; it had been silently broken. Intentional
  replacement is now opt-in via `override=True`.
- **`SemanticBoundaryChunker`:** raw `\r\n` input is now normalised to
  `\n` so paragraph breaks (`\r\n\r\n`) are recognised even when the
  corpus was read in binary mode.
- **`diff_runs` float boundary:** exact-threshold drops (e.g. 0.80 → 0.78
  against a 0.02 threshold) are now classified as `flat` instead of
  `degraded`. IEEE 754 subtraction imprecision had made the classification
  order-dependent. Deltas are rounded to 6 decimals before comparison,
  matching the serialised form.
- **Report path normalisation:** Markdown and HTML reports now render
  filesystem paths with forward slashes so reports generated on Windows
  and Linux are byte-identical.
- **README quickstart:** two copy-pasteable examples used wrong parameter
  names (`window_tokens`/`stride_tokens` instead of `size`/`stride`, and
  `overlap_chars` which doesn't exist on `SemanticBoundaryChunker`). Both
  now use the real chunker signatures and are covered by regression tests.

### Changed

- Test count: 342 → 353 (11 regression tests added for the fixes above).

## [1.0.2] - 2026-04-16

### Fixed (Phase 3 Eval Cycle B — 7 bugs)

- **`SemanticBoundaryChunker` offset corruption (HIGH):** Cycle A's CRLF
  fix (`\r\n` → `\n` rebinding) left chunk `start`/`end` offsets in
  normalised-text space, so `original[ch.start:ch.end] != ch.text`
  whenever the input contained CRLF. `StructuralMarkdownChunker`
  inherited the bug because it composes the inner chunker. Fixed by
  detecting paragraphs with a CRLF-aware regex (no rebinding) so offsets
  remain valid in the original text.
- **Duplicate `TestSemanticBoundaryChunker` class (HIGH):** Cycle A
  appended a second class with the same name to `tests/test_chunkers.py`,
  silently dropping 5 of the 6 original SemanticBoundaryChunker tests
  (Python class redefinition replaces the prior binding). Merged the two
  classes so all tests run.
- **README CLI documentation drift (HIGH):** README documented several
  flags that don't exist (`--fail-on-degraded`, `--top-k`, `--n`,
  `--run`, `--thresholds` plural, `--format markdown`). Replaced with
  the actual CLI surface (`--no-fail`, `--k`, `--questions`, `--in`,
  `--threshold` repeatable, `--format md`).
- **README programmatic API drift (HIGH):** the example used
  `RunConfig(top_k=10)` (wrong field — it's `top_k_cap`) and
  `diff.any_degraded` (no such attribute — use the `diff.degraded` list).
  Both updated and re-verified end-to-end.
- **NaN/Inf summary leak in HTML/Markdown (MEDIUM):** Cycle A fixed the
  per-query `None` handling but missed NaN/±Inf in the summary path —
  they leaked as Python's `nan`/`inf` literals instead of `null`. Added
  `_metric_cell` helper that handles both None and non-finite floats
  uniformly across summary and per-query rendering.
- **Test-suite ruff hygiene (MEDIUM):** the project's pyproject configures
  ruff to scan the whole tree (`select = ["E","F","W","I","B"]`), but the
  test files had 27 unresolved violations (unused imports, `B017` blind
  `Exception`, `E741` ambiguous `l`, `F811` class redefinition, `F841`
  unused locals). All resolved. Blind `pytest.raises(Exception)` now
  asserts the specific `dataclasses.FrozenInstanceError`.
- **`_cmd_diff` triple-render (LOW):** the diff CLI rendered the same
  Markdown three times when neither `--out` nor `--markdown` was set.
  Now renders once and reuses the string.

### Changed

- Test count: 353 → 362 (1 new regression test for the CRLF-offset bug;
  +5 from the duplicate-class fix that restored silently-dropped tests;
  +3 new regression tests covering NaN/Inf rendering in the summary path
  for both Markdown and HTML, plus a per-query non-finite cross-check).
- `ragcheck/report.py` introduces `_metric_cell` and `_is_null_float`
  helpers; the public renderers `render_html` and `render_markdown`
  retain their signatures.

## [1.0.3] - 2026-04-16

### Fixed (Phase 3 Eval Cycle C — 4 bugs)

- **`load_corpus` walks into hidden directories (HIGH):** the corpus
  loader skipped hidden *files* but not hidden *directories*. Calling
  `ragcheck run --corpus .` from a repo root would silently ingest
  `.git/HEAD`, any `*.txt`/`*.md` files under `.venv/`, build caches,
  etc. — leaking sensitive content into the corpus and inflating
  diagnostics. Now any path component that starts with `.` is skipped at
  any depth. Two new regression tests (`test_ignores_hidden_directories`,
  `test_ignores_nested_hidden_directories`) lock the contract.
- **README `register_chunker` example was broken (HIGH):** the README
  showed a free function `def my_chunker(doc_id, text)` and claimed "any
  callable accepting `(doc_id, text)` works". Reality: `get_chunker`
  *constructs* the registered factory with kwargs, so a free function
  crashes immediately with `TypeError: my_chunker() missing 2 required
  positional arguments`. Updated the README to a class-based example
  (with `__init__` and `chunk` method) that mirrors the actual contract,
  plus a regression test
  (`test_readme_register_chunker_example_actually_works`).
- **`NumpyEmbedder` silently accepted 2D arrays as 1D vectors (MEDIUM):**
  `NumpyEmbedder({"a": np.array([[1,2,3],[4,5,6]])})` used to set
  `dim = shape[0]` (2 — the row count!) and then crash much later inside
  `embed()` with `ValueError: could not broadcast input array from
  shape (6,) into shape (2,)`. Now rejected at construction with a clear
  message; same enforcement when a later mapping entry has the wrong ndim
  or a zero dim. Three regression tests added.
- **`NumpyEmbedder.from_directory` exploded on 1D `vectors.npy` (MEDIUM):**
  a 1D `vectors.npy` would crash with `IndexError: tuple index out of
  range` deep inside `__init__`. Now the from-directory loader checks
  `vectors.ndim == 2` up front and raises a clear ValueError pointing at
  the actual shape. Two regression tests cover 1D and 3D inputs.

### Changed

- Test count: 362 → 370 (+8 regression tests: 2 corpus hidden-directory,
  3 NumpyEmbedder shape-validation in mapping mode, 2 NumpyEmbedder
  shape-validation in from-directory mode, 1 README chunker example).
- `ragcheck/corpus.py::load_corpus` now skips files whose relative path
  contains any dot-prefixed segment.
- `ragcheck/embedders.py::NumpyEmbedder.__init__` and
  `NumpyEmbedder.from_directory` now validate `ndim` up front.
- `README.md` "Register your own" example replaced with a class-based
  factory that round-trips through `register_chunker` + `get_chunker`.

## [1.0.5] - 2026-04-16

### Fixed (Phase 3 Eval Cycle E — 6 bugs)

- **`OpenAIEmbedder` cache corruption crash (HIGH):** `_load_cached`
  decoded cached JSON without validation. A truncated write, a manually
  edited cache file, or a file written by a prior run with a different
  `dim` would raise deep inside `np.asarray` (or succeed and return a
  vector of the wrong shape), crashing every subsequent `embed_many`
  call for that text. The cache is now validated on read: missing
  `vector` key, wrong ndim, or wrong dim all trigger `_cache_path.unlink()`
  and fall through to a live API call. `OSError`/`ValueError`/`TypeError`/
  `JSONDecodeError` are all caught uniformly. Six regression tests
  (`TestOpenAICacheFaultInjectionCycleE`) inject truncated JSON, wrong
  dim, wrong ndim, non-dict payloads, empty files, and bare strings.
- **`StructuralMarkdownChunker` offset invariant violation (HIGH):**
  section text was `.rstrip()`ped before chunking while `section_end`
  still pointed at the un-stripped end of the section, so
  `text[chunk.start:chunk.end] != chunk.text` whenever a section ended
  with trailing whitespace. Fixed by shrinking `section_end` to match
  the rstripped length before constructing chunks, preserving the
  contract that chunk offsets can always be used to recover the original
  text slice. Four regression tests
  (`TestStructuralMarkdownOffsetInvariantCycleE`) cover trailing
  whitespace, multiple sections each with trailing whitespace, mixed
  CRLF+trailing, and a property-style invariant check.
- **`ndcg_at_k`/`dcg_at_k` silently accept NaN/Inf gains (HIGH):** the
  existing `< 0` rejection could not catch NaN (NaN comparisons return
  `False`) or `+Inf` (valid comparison, but propagates to `sum` →
  infinite DCG → `nan` nDCG after division). A corrupted gold set or a
  library caller passing `float("nan")` as a relevance would silently
  poison summary metrics. Added explicit `math.isnan`/`math.isinf`
  checks before the sign check in both `dcg_at_k` and `ndcg_at_k`
  (ideal side as well). Seven regression tests
  (`TestRelevanceFiniteRejectionCycleE`) cover NaN, +Inf, -Inf in both
  functions plus a round-trip through `aggregate_metric`.
- **`load_gold_set` accepts NaN/Inf relevance in JSON (MEDIUM):** the
  Python stdlib json module accepts `NaN`, `Infinity`, `-Infinity` as
  number literals by default. A gold file containing
  `"relevance": {"doc-1": NaN}` would load silently and then crash
  nDCG inside `aggregate_metric`. Now rejected at load time with a
  clear error pointing at the offending key. Four regression tests
  (`TestLoadGoldSetFiniteRelevanceCycleE`).
- **`diff_runs` silently accepts incompatible `top_k_cap` (MEDIUM):**
  comparing a run with `top_k_cap=5` against one with `top_k_cap=10`
  is meaningless for `recall@10` — the first run literally could not
  have observed any retrievals in positions 6-10, so the "degraded"
  signal is artificial. The diff now emits a warning (visible in both
  the JSON `warnings` array and the rendered Markdown under a
  `## Warnings` section) whenever `top_k_cap` or `k_values` differ
  between baseline and head. The warning does not change the exit
  code — it's a legitimate comparison — but it's now impossible to miss.
  Covered by `TestDiffRunsTopKCapWarningCycleE`.
- **`save_gold_set` leaks native path separators (MEDIUM):** gold entries
  with a `source` like `docs\\part1\\intro.md` were serialised as-is,
  breaking byte-identical diffs between Windows and POSIX gold files and
  round-tripping unexpectedly through `load_gold_set`. Now normalised to
  forward slashes on write, matching `load_corpus` and
  `render_diff_markdown`. Three regression tests
  (`TestSaveGoldSetSourceNormalizationCycleE`).

### Changed

- Test count: 390 → 420 (+30 regression tests: 6 OpenAIEmbedder cache
  fault injection, 4 StructuralMarkdownChunker offset invariant, 7 DCG/nDCG
  finite-relevance rejection, 4 load_gold_set NaN relevance, 3 save_gold_set
  path normalisation, 6 diff_runs top_k_cap/k_values warning).
- `ragcheck/embedders.py::OpenAIEmbedder._load_cached` now validates
  cached payload shape and deletes corrupted cache entries on read.
- `ragcheck/chunkers.py::StructuralMarkdownChunker.chunk` now shrinks
  `section_end` to match the rstripped section text, preserving the
  `text[start:end] == chunk.text` invariant.
- `ragcheck/metrics.py::dcg_at_k` and `ndcg_at_k` reject NaN/Inf gains
  and relevance values before the sign check.
- `ragcheck/corpus.py::load_gold_set` rejects non-finite relevance
  values at load time; `save_gold_set` now normalises `source` paths to
  forward slashes.
- `ragcheck/diff.py::DiffResult` gains a `warnings` field populated
  whenever baseline and head disagree on `top_k_cap` or `k_values`;
  rendered in both JSON and Markdown output.

## [1.0.4] - 2026-04-16

### Fixed (Phase 3 Eval Cycle D — 6 bugs)

- **`load_corpus` symlink/junction escape (HIGH):** the corpus loader
  resolved `Path.iterdir()` results without checking whether the resolved
  target still lived under the corpus root. A symlink or Windows
  directory junction planted inside the corpus could redirect into
  `/etc`, `C:\Windows`, a sibling project's secrets, etc., and everything
  reachable from there would be ingested as "corpus content". Fixed by
  calling `root.resolve()` once and checking every candidate file with
  `file.resolve().relative_to(root_resolved)`; anything escaping the
  root is silently skipped. Three regression tests
  (`TestLoadCorpusSymlinkSafetyCycleD`) cover POSIX symlinks, Windows
  junctions, and internal symlinks (which must still be allowed).
- **`ndcg_at_k` can exceed 1.0 on duplicated retrievals (HIGH):** if the
  same relevant doc appears multiple times in the retrieved list, the
  naive DCG loop credits each hit independently while the ideal DCG
  assumes each relevant doc is counted once. `ndcg_at_k(["a","a","b"],
  {"a": 1.0})` used to return `> 1.0`, which is mathematically
  impossible and broke threshold comparisons for summary metrics.
  Retrieved list is now deduped on-the-fly: duplicates contribute zero
  gain. Six regression tests (`TestNdcgDuplicatesCycleD`) including a
  property-style upper-bound check `0 ≤ ndcg@k ≤ 1` over random
  permutations with duplicates.
- **`ragcheck synth` crashes on non-UTF-8 consoles (HIGH):** the NOTE
  line used a Unicode em-dash (`\u2014`), which is not representable in
  cp949 (Korean), cp932 (Japanese), or cp1252 (Windows Western) default
  consoles. The entire `synth` subcommand crashed mid-run with
  `UnicodeEncodeError` for Korean and Japanese users. Replaced the
  em-dash with a plain ASCII hyphen and added two regression tests:
  one simulates cp949 via `TextIOWrapper(encoding="cp949",
  errors="strict")` and confirms no crash, the other asserts every byte
  emitted by synth is pure ASCII.
- **Empty corpus directory silently returns zero metrics (MEDIUM):**
  `run_evaluation(config=RunConfig(corpus_path="empty/"))` used to load
  nothing from disk, then run every metric against the empty corpus and
  report `recall@5 = 0.0` across the board. Users misconfiguring their
  `--corpus` flag saw a clean green "zero hits" run instead of an
  actionable error. Now raises `ValueError("corpus is empty: no .txt/.md
  files found under ...")` when the path was loaded from disk and
  produced an empty list. The in-memory `corpus=[]` opt-in path (used
  by tests and library callers) is preserved.
- **Test-count drift guard was too weak (MEDIUM):** `test_doc_drift.py`
  only checked that `N tests` appeared somewhere in the README. The
  shields.io badge uses `%20` instead of a literal space, so the badge
  could drift independently and users would see a stale count on
  GitHub. CHANGELOG wasn't checked at all. Added three stricter guards:
  `test_readme_badge_count_matches_collected` (exact
  `tests-N%20passing-brightgreen` pattern),
  `test_readme_sentence_count_matches_collected` (exact
  `**N tests. Offline` pattern), and
  `test_changelog_references_current_count` (the current count must
  appear in `CHANGELOG.md`).
- **JSON config block uses Windows-native backslashes (MEDIUM):**
  `RunConfig(corpus_path=r"a\b\c").to_dict()` serialised
  `"corpus_path": "a\\b\\c"`, breaking byte-identical diffing across
  Windows/POSIX and polluting comparison reports. Fixed by normalising
  every path-shaped field in `to_dict()` to forward slashes (matching
  `render_diff_markdown` + `render_report_markdown` which were already
  doing this). Three regression tests including an absolute
  `C:\Users\alice\data\corpus` → `C:/Users/alice/data/corpus` check.

### Changed

- Test count: 370 → 390 (+20 regression tests: 3 corpus symlink-safety,
  6 nDCG duplicate-robustness, 2 synth cp949 safety, 3 empty-corpus
  error, 3 stricter doc-drift, 3 JSON path-normalisation).
- `ragcheck/corpus.py::load_corpus` now resolves the root once and
  `relative_to`-checks every candidate file against it.
- `ragcheck/metrics.py::ndcg_at_k` dedupes the retrieved list while
  walking it so repeated ids contribute zero gain, guaranteeing
  `0 ≤ ndcg ≤ 1`.
- `ragcheck/cli.py::_cmd_synth` NOTE line is now pure ASCII.
- `ragcheck/runner.py::run_evaluation` raises on empty filesystem
  corpora; programmatic `corpus=[]` still works.
- `ragcheck/runner.py::RunConfig.to_dict` normalises `corpus_path` and
  `gold_path` to forward slashes for cross-platform byte-identical
  output.
- `tests/test_doc_drift.py` now pins the shields.io badge, the
  README tagline sentence, and the CHANGELOG entry to the current
  pytest-collected count.
