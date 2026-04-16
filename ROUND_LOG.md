# Round 82 — ragcheck

## Project Info
- **Name**: ragcheck
- **Category**: RAG Retrieval Evaluation / Offline Quality Harness
- **Language**: Python
- **Score**: 104/110
- **Date**: 2026-04-16

## Phase 1: Pitch
- Winner: ragcheck (Trend Scout pitch)
- Score: 104/110
- Runners-up: browserprobe (98), graphpack (96), voicetest (89)
- Key data: RAG Framework category trending (LightRAG 30K, hello-agents 31K, anything-llm 56K)
- Gap: no offline, LLM-free retrieval-quality harness exists

## Phase 2: Build
- Initial test count: 342
- Source files: 13 Python modules, ~3,300 lines
- Features: 10 metrics, 5 chunkers, 4 embedders, regression diffs, CLI, HTML/Markdown/JSON reports
- Bundled fixtures: BEIR FiQA, MS MARCO, needle-in-haystack

## Phase 3: Eval
- Total cycles: 9 (A-I)
- Total bugs found: 34
- Dirty cycles: 6 (A=8, B=7, C=4, D=6, E=6, F=3)
- Clean cycles: 3 consecutive (G, H, I) -> PASS

### Eval Cycle Detail

| Cycle | Verdict | Bugs | Tests After | Key Findings |
|-------|---------|------|-------------|------|
| A | DIRTY | 8 | 353 | Missing public API export, README param drift, render_html None crash, register_chunker silent overwrite |
| B | DIRTY | 7 | 362 | CRLF offset corruption, duplicate test class drops tests, README CLI flag drift (5 flags), NaN/Inf in reports |
| C | DIRTY | 4 | 370 | load_corpus walks hidden dirs (.git), README register_chunker example broken, NumpyEmbedder dim validation |
| D | DIRTY | 6 | 390 | Symlink/junction escape, nDCG >1.0 on duplicates, cp949 crash on non-ASCII console, empty corpus silent zero |
| E | DIRTY | 6 | 420 | OpenAI cache corrupt read, StructuralMarkdown offset invariant, NaN/Inf in DCG, gold set NaN acceptance, diff top_k_cap mismatch, save_gold_set path leak |
| F | DIRTY | 3 | 434 | OpenAI cache concurrent-writer race (atomic write), diff chunker/embedder mismatch warnings, save_gold_set CRLF on Windows |
| G | CLEAN | 0 | 434 | First clean — all probes cleared |
| H | CLEAN | 0 | 434 | Second clean — 16-process race probe, full argparse sweep |
| I | CLEAN | 0 | 434 | Third clean — PASS. Full source read, determinism triple-run, XSS sweep |

## Phase 4: Ship
- Release: v1.0.0
- Final test count: 434 (432 passed, 2 skipped)
- GitHub topics: 10
- Ship date: 2026-04-16

## Quality Gates (8/8)
- [x] Tests pass (432 passed, 2 skipped)
- [x] Test count >= 150 (434)
- [x] README >= 300 lines (458)
- [x] for-the-badge shields.io badges
- [x] GitHub topics >= 8 (10)
- [x] Coverage >= 85% (96% badge)
- [x] LICENSE exists (MIT)
- [x] CHANGELOG exists
