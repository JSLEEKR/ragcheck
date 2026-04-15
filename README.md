# ragcheck

[![tests](https://img.shields.io/badge/tests-390%20passing-brightgreen?style=for-the-badge)](https://github.com/JSLEEKR/ragcheck/actions)
[![coverage](https://img.shields.io/badge/coverage-96%25-brightgreen?style=for-the-badge)](https://github.com/JSLEEKR/ragcheck)
[![python](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge)](https://python.org)
[![license](https://img.shields.io/badge/license-MIT-lightgrey?style=for-the-badge)](LICENSE)
[![llm-free](https://img.shields.io/badge/LLM--as--judge-not%20required-purple?style=for-the-badge)](#why-this-exists)
[![deterministic](https://img.shields.io/badge/output-byte%20identical-blue?style=for-the-badge)](#determinism-contract)
[![offline](https://img.shields.io/badge/runtime-offline-green?style=for-the-badge)](#installation)
[![bench](https://img.shields.io/badge/bench-%3C1s-brightgreen?style=for-the-badge)](#benchmark)

> **390 tests. Offline retrieval-quality harness for RAG systems. No LLM-as-judge.**
>
> `recall@k`, `precision@k`, `hit_rate@k`, `MRR`, `nDCG` (binary + graded),
> `context_precision`, `context_recall`, plus chunking diagnostics and
> regression diffs — all deterministic, all byte-identical across runs,
> all free of LLM calls.

---

## Why This Exists

Everyone building a RAG system has the same debugging loop:

1. Swap chunker → retrieval feels worse.
2. Try a different embedder → ???
3. Add reranker → ???
4. Deploy → user complains 🎉.

The loop is broken because teams never **measure** retrieval. They measure
end-to-end answer quality with an LLM-as-judge, and by then the signal from
retrieval is buried under 200B+ generator parameters.

`ragcheck` fixes this the boring way: **standard IR metrics, computed
offline on a gold set, with deterministic output you can diff in CI.**

### What this tool does NOT do

- It does **not** call an LLM to judge answers. That's `ragas`, `deepeval`,
  `trulens`, and friends. They are great for end-to-end pipelines. They are
  the wrong tool for "did my chunker regress?"
- It does **not** spin up a vector DB. You bring your corpus as plain text
  files; `ragcheck` chunks + embeds + scores in-process.
- It does **not** require a GPU. The default `HashEmbedder` runs a full
  bench in **under a second** on CPU, so you can gate PRs on it.

### What this tool DOES do

- Computes **10 retrieval metrics** per query, aggregates over the gold set.
- Ships **5 chunkers** and **3 embedders** (plus `HashEmbedder` for CI).
- Ships **3 fixtures** (BEIR FiQA subset, MS MARCO subset, synthetic
  needle-in-haystack) so you can start evaluating in 5 seconds.
- Detects **regressions between runs** with configurable thresholds and
  exits non-zero when any metric drops below them — drop it in CI.
- Generates **byte-identical** JSON output across runs on the same inputs.
  `diff` returns empty. Commits are meaningful.

---

## Installation

```bash
pip install ragcheck
# optional: use heavy embedders
pip install "ragcheck[sentence-transformers]"
pip install "ragcheck[openai]"
```

Requires Python 3.9+. Core dependencies: `numpy`, `jinja2`. No network,
no GPU, no LLM required for any of the default flows.

---

## 60-Second Quickstart

```bash
# 1. Evaluate the bundled BEIR fixture
ragcheck run \
  --corpus $(python -c 'import ragcheck.fixtures as f; print(f.BEIR_FIQA_DIR / "corpus")') \
  --gold   $(python -c 'import ragcheck.fixtures as f; print(f.BEIR_FIQA_DIR / "gold.json")') \
  --out runs/baseline.json

# 2. Change chunker, re-run
ragcheck run --corpus ... --gold ... --out runs/candidate.json \
  --chunker sliding-window --chunker-args '{"size": 120, "stride": 60}'

# 3. Diff
ragcheck diff runs/baseline.json runs/candidate.json
# Exit 1 if any metric drops more than the threshold (default 0.02).
# Pass --no-fail to always exit 0 (e.g. for advisory CI checks).
```

---

## CLI

```
ragcheck run       --corpus DIR --gold FILE [--chunker NAME] [--embedder NAME] --out FILE
ragcheck diff      BASELINE.json HEAD.json [--threshold KEY=VAL ...] [--no-fail]
ragcheck bench     [--json]
ragcheck synth     --corpus DIR --out FILE [--questions 50] [--seed 42]
ragcheck report    --in FILE --format {json,md,html} [--out FILE]
ragcheck chunkers  list
ragcheck embedders list
```

### `ragcheck run`

Load corpus + gold set, chunk, embed, retrieve top-k per query, compute
all metrics, dump deterministic JSON.

```bash
ragcheck run \
  --corpus ./docs \
  --gold   ./eval/gold.json \
  --chunker semantic-boundary \
  --chunker-args '{"max_chars": 1200}' \
  --embedder hash \
  --k 1,3,5,10 \
  --out runs/2026-04-16.json
```

### `ragcheck diff`

Compute per-metric deltas between two runs. Exits 1 if any metric drops
more than its threshold. Default threshold is **0.02** for every metric;
override per-metric-prefix by repeating `--threshold KEY=VAL`. Pass
`--no-fail` to always exit 0 (e.g. advisory CI).

```bash
ragcheck diff runs/baseline.json runs/candidate.json \
  --threshold recall@5=0.01 --threshold mrr=0.03
```

### `ragcheck bench`

Runs the three bundled fixtures end-to-end and prints a timing table.
**Completes in under a second on CPU.** Use it to smoke-test after install:

```
fixture          | docs | queries | recall@5 | mrr      | ndcg@5   | seconds
-----------------|------|---------|----------|----------|----------|--------
beir_fiqa        |    8 |      10 |   0.5500 |   0.8083 |   0.5463 |  0.006
ms_marco         |    6 |       8 |   0.7500 |   0.8438 |   0.7193 |  0.004
needle_haystack  |   78 |       6 |   0.8333 |   0.6444 |   0.6696 |  0.032
-----------------|------|---------|----------|----------|----------|--------
total elapsed: 0.042s
```

### `ragcheck synth`

Generate a synthetic gold set from your corpus when you don't have labels yet.
Picks the most distinctive sentence from each document and rephrases it into
a question. Deterministic given the same `--seed`.

### `ragcheck report`

Re-render a saved run as Markdown or HTML (JSON is the input format).
Useful for pinning in pull-request descriptions.

### `ragcheck chunkers list` / `ragcheck embedders list`

Print the registered chunkers / embedders. Output stays in sync with the
source — fail fast if someone registers a duplicate.

---

## Metrics

All metrics are **deterministic, pure functions** (no state, no random
seeds, no LLM calls). Each is implemented in one place and hand-verified
against an independent textbook calculation in `tests/test_metrics.py`.

| Metric                  | Formula                                                                 | Notes |
|-------------------------|-------------------------------------------------------------------------|-------|
| `recall@k`              | distinct relevant in top-k / total relevant                             | deduplicated by design |
| `precision@k`           | hits in top-k / k                                                       | denominator is k, not retrieved length |
| `hit_rate@k`            | 1.0 iff any top-k item is relevant                                      | aka success@k |
| `mrr`                   | 1 / rank of first relevant item                                         | 0.0 if no relevant item retrieved |
| `dcg@k` / `ndcg@k`      | Σ (2^gain − 1) / log₂(i+1), normalised by ideal DCG                    | binary or graded relevance |
| `context_precision`     | LangChain/Ragas-style average precision@i at hit positions              | rewards putting relevance at top |
| `context_recall`        | recall over the retrieved window                                        | equivalent to recall@len(retrieved) |
| `f1@k`                  | 2·p·r / (p+r)                                                           | 0.0 when both zero |
| `average_precision`     | MAP's per-query component                                               | aggregated by caller |

Default k-values are **{1, 3, 5, 10}**, matching common IR benchmarks.
Override with `--k 1,5,20,100`.

---

## Chunkers (5 built-in)

Every chunker produces `Chunk(id, doc_id, start, end, text, metadata)`
where `id = sha1("doc_id|start|end|flavor")`. **Deterministic, content-addressed.**

| Chunker                   | When to use                                                            |
|---------------------------|------------------------------------------------------------------------|
| `fixed-token`             | Baseline; reproducible; no dependencies on text structure              |
| `sliding-window`          | When you care about edge cases falling on chunk boundaries             |
| `sentence`                | Short docs, FAQ-style corpora                                          |
| `semantic-boundary`       | Long docs where paragraphs are the natural unit                        |
| `structural-markdown`     | Docs / READMEs / wikis with `# headings` and code blocks               |

Register your own:

```python
from ragcheck.chunkers import register_chunker, Chunk

class MyChunker:
    name = "my-chunker"

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        return [Chunk(doc_id=doc_id, chunk_id="x", text=text, start=0, end=len(text))]

register_chunker("my-chunker", MyChunker)
# get_chunker("my-chunker", max_chars=1200) constructs and returns an instance.
```

`register_chunker` takes a **factory** (typically a class) that accepts
keyword arguments and returns a chunker instance with a
`.chunk(doc_id: str, text: str) -> list[Chunk]` method. Pass
`override=True` to intentionally replace an existing factory; otherwise
duplicate registrations raise `ValueError` to fail fast.

---

## Embedders (3 built-in + Hash)

| Embedder                | Install with              | Notes                                              |
|-------------------------|---------------------------|----------------------------------------------------|
| `hash`                  | (built in)                | Deterministic md5 shingle hashing; for CI / bench |
| `sentence-transformers` | `ragcheck[sentence-transformers]` | `all-MiniLM-L6-v2` default; lazy imported   |
| `openai`                | `ragcheck[openai]`        | Disk-cached by sha1(model+text); lazy imported     |
| `numpy`                 | (built in)                | Read pre-computed vectors (`vectors.npy`+`ids.json`) |

`HashEmbedder` produces L2-normalised 128-dimensional vectors from n-gram
shingles. It's **not good at semantics** — that's intentional. It gives you a
stable, offline baseline so you can isolate changes from your real embedder.

---

## Chunking Diagnostics

`ragcheck` also answers the question "is my chunker doing something insane?"
by computing structural diagnostics on every run:

| Diagnostic       | What it tells you                                                      |
|------------------|------------------------------------------------------------------------|
| `coverage`       | Fraction of every source document covered by at least one chunk       |
| `duplicate_ratio`| Fraction of chunks that share the exact same content hash              |
| `orphan_chunks`  | Chunks that never appear in any query's top-k (potential dead weight)  |
| `size_histogram` | Distribution of chunk character lengths across 7 buckets               |

Example (BEIR FiQA with `fixed-token`, `tokens_per_chunk=40`):

```
0-49       |
50-99      | 3
100-199    | ████████████████████ 17
200-499    | ██ 2
500-999    |
1000-1999  |
2000+      |
```

---

## Regression Diff

`ragcheck diff` reports per-metric deltas with a status flag:

- `improved`  — metric increased by at least its threshold
- `flat`      — change within ±threshold
- `degraded`  — metric dropped by more than its threshold
- `new`       — metric exists in head but not baseline
- `removed`   — metric exists in baseline but not head

Default threshold is `0.02` (2 points) for every metric.

```bash
# Fail CI if recall@5 drops by more than 1 point, but tolerate 5-point changes
# on the slow-moving context_recall
ragcheck diff base.json head.json \
  --threshold recall@5=0.01 \
  --threshold context_recall=0.05
```

---

## Determinism Contract

> Two runs of `ragcheck run` over the same corpus + gold set + config MUST
> produce byte-identical JSON.

This is enforced by:

- Chunk IDs from `sha1(doc_id + start + end + flavor)`
- L2-normalised embeddings with stable-sort top-k
  (`numpy.argsort(..., kind="stable")`)
- `json.dumps(sort_keys=True, ensure_ascii=False)`
- 6-decimal-place float formatting via `format_float`
- **No timestamps** in run output (add-only via `--embed-timestamp`)
- `corpus_sha1` stamped in output so `diff` warns when the corpus changes
- No `set` iteration order leaks into output

Test `test_runner.py::TestDeterminism::test_two_runs_byte_identical`
enforces this on every CI build.

---

## Comparison

| Tool                | LLM-as-judge required | Retrieval metrics | Chunking diagnostics | Regression diff | Byte-identical output | Runs offline | Runtime (bench) |
|---------------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **ragcheck**        | ❌ | ✅ 10 metrics | ✅ coverage, dup, orphans, sizes | ✅ per-metric thresholds | ✅ | ✅ | **<1s** |
| ragas               | ✅ | partial | ❌ | ❌ | ❌ | ❌ | ~minutes |
| deepeval            | ✅ | partial | ❌ | ❌ | ❌ | ❌ | ~minutes |
| BenchmarkQED        | ✅ | partial | ❌ | ❌ | ❌ | ❌ | ~minutes |
| trulens             | ✅ | partial | ❌ | ❌ | ❌ | ❌ | ~seconds-minutes |

`ragcheck` does not replace these tools. It sits **under** them: it catches
retrieval regressions before you spend LLM quota evaluating generations on
top of bad context.

---

## Architecture

```
+-----------+     +----------+     +-----------+     +-----------+
|  Corpus   | --> | Chunker  | --> | Embedder  | --> | Retrieval |
| (text/md) |     |   (5)    |     |    (4)    |     |   top-k   |
+-----------+     +----------+     +-----------+     +-----------+
                                                           |
                                                           v
                                                    +-------------+
                                                    |   Metrics   |
                                                    | (pure fns)  |
                                                    +-------------+
                                                           |
                                                           v
                                   +-----------------+ +---+---+ +------+
                                   |  Diagnostics    | | JSON  | | Diff |
                                   | (coverage, dup) | |(byte= )| | (±δ) |
                                   +-----------------+ +-------+ +------+
```

All layers are swappable; none depend on network or LLMs.

---

## Where this fits in my stack

`ragcheck` complements four sibling projects:

- **[ctxpack](https://github.com/JSLEEKR/ctxpack)** — compresses context
  windows before retrieval. `ragcheck` measures whether that compression
  preserves retrieval quality.
- **[ctxlens](https://github.com/JSLEEKR/ctxlens)** — introspects context
  window utilisation at runtime. `ragcheck` proves what goes in the window.
- **[promptdiff](https://github.com/JSLEEKR/promptdiff)** — regression
  tests for prompts. `ragcheck` does the same for retrieval, so prompt
  tests don't mask chunking bugs.
- **[mocklm](https://github.com/JSLEEKR/mocklm)** — deterministic LLM
  mocking. `ragcheck` runs without any LLM at all; together they give you a
  fully offline RAG regression harness.

Each of these stands alone. Together they form the CI substrate for an
agent shop that treats retrieval, context, prompts, and generation as
independent components you can regress on separately.

---

## Programmatic API

Every CLI operation is a pure Python function:

```python
import ragcheck
from ragcheck import RunConfig, run_evaluation, diff_runs

result = run_evaluation(config=RunConfig(
    corpus_path="./docs",
    gold_path="./eval/gold.json",
    chunker="semantic-boundary",
    embedder="hash",
    top_k_cap=20,
))

print(result.summary["recall@5"])
print(result.diagnostics["coverage"])

# Load two saved runs and diff
diff = diff_runs(
    ragcheck.load_result_json("runs/base.json"),
    ragcheck.load_result_json("runs/head.json"),
    fail_on_degraded=True,
)
if diff.degraded:
    raise SystemExit(1)
```

---

## Bundled Fixtures

- **`beir_fiqa`** — 8 finance-question documents + 10 queries drawn from the
  BEIR FiQA benchmark style (no external download; subset bundled).
- **`ms_marco`** — 6 passages + 8 queries in MS MARCO's passage-ranking style.
- **`needle_haystack`** — synthetic needle-in-haystack with 6 needles and 72
  distractor haystack documents. Good for stress-testing chunker boundary
  behaviour without leaking real labels.

All three ship in `ragcheck.fixtures`; `bench` runs all three.

---

## Limitations

- **No reranker evaluation.** If your pipeline has a cross-encoder reranker,
  run it to produce the "retrieved" list and pass that list into `ragcheck`.
- **Binary relevance by default.** Graded relevance works via the
  `relevance` mapping in the gold file (`{doc_id: 1.0, doc_id2: 2.5, ...}`),
  but most `ragcheck` commands optimise for the binary case.
- **No latency simulation.** `elapsed_seconds` in bench is wall-clock for the
  retrieval itself, not a realistic p99 — measure that at your serving tier.
- **Corpus hashing is content-level, not metadata-level.** Renaming a file
  changes the hash; editing one character does too. That is intentional
  (diffs should warn) but worth knowing.
- **No built-in multi-query fusion.** If you use RAG-fusion or HyDE, run
  `ragcheck` on each sub-query and aggregate externally.

---

## Development

```bash
git clone https://github.com/JSLEEKR/ragcheck.git
cd ragcheck
pip install -e ".[dev,sentence-transformers,openai]"
pytest           # 390 tests pass (doc-drift guard included)
ruff check ragcheck/
mypy ragcheck/
python -m ragcheck bench
```

Contributions welcome. All new metrics need a textbook-verified test.
All new chunkers need coverage + diagnostics smoke tests. All public API
changes need a `CHANGELOG.md` entry.

---

## License

[MIT](LICENSE) © 2026 JSLEEKR
