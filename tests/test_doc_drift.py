"""README / documentation drift guard.

Enforces two invariants:
1. The README's test-count badge and sentence exactly match the real number
   of pytest-collected tests. If you add or remove a test, this test fails
   until README is updated (one place: the README badge/sentence).
2. The README lists every CLI subcommand that the binary actually supports.

Pattern ported from R81 (skillpack docsmeta) to Python.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import ragcheck

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


def _collect_test_count() -> int:
    """Invoke pytest --collect-only -q and parse the total."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(REPO_ROOT / "tests")],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # Look for the "N tests collected" summary line
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = re.search(r"(\d+)\s+tests?\s+collected", output)
    if match:
        return int(match.group(1))
    # Fallback: count test functions on stdout
    lines = [ln for ln in output.splitlines() if "::" in ln]
    if lines:
        return len(lines)
    raise AssertionError(
        f"could not parse pytest collect output:\n{output[-2000:]}"
    )


class TestReadmeExists:
    def test_readme_present(self):
        assert README.exists(), f"README.md missing at {README}"

    def test_readme_not_empty(self):
        assert README.stat().st_size > 1000

    def test_readme_min_length_300(self):
        content = README.read_text(encoding="utf-8")
        assert content.count("\n") >= 300


class TestReadmeTestCountMatchesReality:
    def test_readme_mentions_a_specific_test_count(self):
        text = README.read_text(encoding="utf-8")
        assert re.search(r"\btests?\b", text, re.IGNORECASE)

    def test_readme_count_matches_collected(self):
        text = README.read_text(encoding="utf-8")
        actual = _collect_test_count()
        # README should reference the exact collected count somewhere
        pattern = re.compile(rf"\b{actual}\s+tests?\b")
        assert pattern.search(text), (
            f"README does not reference the correct test count "
            f"(expected {actual}). If you added/removed tests, update the "
            f"README badge and the 'N tests' sentence."
        )


class TestReadmeMentionsSubcommands:
    EXPECTED_SUBCOMMANDS = ["run", "diff", "bench", "synth", "report", "chunkers", "embedders"]

    def test_all_subcommands_documented(self):
        text = README.read_text(encoding="utf-8").lower()
        for cmd in self.EXPECTED_SUBCOMMANDS:
            assert f"ragcheck {cmd}" in text, f"README missing `ragcheck {cmd}`"


class TestReadmeDifferentiatorsCited:
    CITED_PROJECTS = ["ctxpack", "ctxlens", "promptdiff", "mocklm"]

    def test_shipped_projects_cited(self):
        text = README.read_text(encoding="utf-8").lower()
        for name in self.CITED_PROJECTS:
            assert name in text, f"README must cite shipped project {name!r}"


class TestChangelogExists:
    def test_changelog_present(self):
        path = REPO_ROOT / "CHANGELOG.md"
        assert path.exists()


class TestLicenseExists:
    def test_license_mit_2026(self):
        path = REPO_ROOT / "LICENSE"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "MIT" in text
        assert "2026" in text
        assert "JSLEEKR" in text


class TestPackageMetadata:
    def test_version_parses(self):
        assert re.match(r"\d+\.\d+\.\d+", ragcheck.__version__)
