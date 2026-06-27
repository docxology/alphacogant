from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript"
SECTION_RE = re.compile(r"^#\s+.+\{#sec:[a-z0-9_-]+\}\s*$")
DISPLAY_MATH_RE = re.compile(r"\$\$\s*\n(.*?)\n\$\$(?:\s*\{#eq:[a-z0-9_-]+\})?", re.DOTALL)
LABELED_EQUATION_RE = re.compile(r"\$\$\s*\n.*?\n\$\$\s*\{#eq:[a-z0-9_-]+\}", re.DOTALL)
CITATION_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
REFERENCE_ENTRY_RE = re.compile(r"^(\d+)\.\s", re.MULTILINE)


def manuscript_markdown() -> list[Path]:
    return sorted(MANUSCRIPT_DIR.glob("[0-9][0-9]_*.md"))


def test_top_level_sections_have_crossref_labels() -> None:
    for path in manuscript_markdown():
        first_heading = next(
            line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("# ")
        )
        assert SECTION_RE.match(first_heading), f"{path.name} lacks a top-level sec label"


def test_manuscript_uses_crossref_equation_contract() -> None:
    for path in manuscript_markdown():
        text = path.read_text(encoding="utf-8")
        assert r"\tag{" not in text, f"{path.name} uses manual equation tags"
        assert r"\begin{equation}" not in text, f"{path.name} uses raw LaTeX equation blocks"
        assert "§" not in text, f"{path.name} uses hardcoded section-sign references"
        display_blocks = DISPLAY_MATH_RE.findall(text)
        labeled_blocks = LABELED_EQUATION_RE.findall(text)
        assert len(display_blocks) == len(labeled_blocks), (
            f"{path.name} has an unlabeled display equation"
        )


def test_cover_art_is_declared_in_manuscript_config() -> None:
    config = (MANUSCRIPT_DIR / "config.yaml").read_text(encoding="utf-8")
    assert 'image: "figures/cover_art.png"' in config


def test_claim_boundary_language_stays_reduced_model_only() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_markdown())
    assert "not financial advice" in text
    assert "does not reproduce AlphaFund's proprietary" in text
    assert re.search(r"same\s+control\s+problem", text) is None
    assert "type checker rejects it" not in text
    assert "type-checker enforces" not in text
    assert "is incapable of expressing the violation" not in text
    assert "cannot occur by construction" not in text
    assert "control problem **is**" not in text
    assert "every allocation is a gradient" not in text
    assert "negative-EFE action value" in text


def test_bootstrap_caption_values_are_token_injected() -> None:
    text = (MANUSCRIPT_DIR / "05_epistemic_and_pragmatic_value.md").read_text(encoding="utf-8")
    assert "{{CREATE_RATE_MEAN}}" in text
    assert "{{DECAY_RATE_MEAN}}" in text
    assert "{{HEADLINE_T_RSI}}" in text
    assert "{{BOOTSTRAP_N}}" in text
    assert "0.1412" not in text
    assert "0.2185" not in text

    limitations = (MANUSCRIPT_DIR / "09_limitations.md").read_text(encoding="utf-8")
    assert "n={{BOOTSTRAP_N}}" in limitations
    assert "n=200" not in limitations


def _defined_reference_ids() -> set[int]:
    refs = (MANUSCRIPT_DIR / "99_references.md").read_text(encoding="utf-8")
    return {int(match.group(1)) for match in REFERENCE_ENTRY_RE.finditer(refs)}


def _used_reference_ids() -> set[int]:
    used: set[int] = set()
    for path in manuscript_markdown():
        text = path.read_text(encoding="utf-8")
        for match in CITATION_RE.finditer(text):
            for raw_id in match.group(1).split(","):
                used.add(int(raw_id.strip()))
    return used


def test_reference_ids_defined_and_used() -> None:
    defined_ids = _defined_reference_ids()
    used_ids = _used_reference_ids()
    undefined_ids = sorted(used_ids - defined_ids)
    unused_ids = sorted(defined_ids - used_ids)
    assert not undefined_ids, f"Manuscript uses undefined references: {undefined_ids}"
    assert not unused_ids, f"References defined but not used in manuscript: {unused_ids}"
