from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = PROJECT_ROOT / "docs" / "manuscript"
DISPLAY_MATH_RE = re.compile(r"\$\$\s*\n(.*?)\n\$\$(?:\s*\{#eq:[a-z0-9_-]+\})?", re.DOTALL)
LABELED_EQUATION_RE = re.compile(r"\$\$\s*\n.*?\n\$\$\s*\{#eq:[a-z0-9_-]+\}", re.DOTALL)
CITATION_KEY_RE = re.compile(r"@(\w+)")
BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
REFERENCES_BIB = MANUSCRIPT_DIR / "references.bib"


def manuscript_markdown() -> list[Path]:
    return sorted(MANUSCRIPT_DIR.glob("[0-9][0-9]_*.md"))


def test_section_references_resolve_to_unique_labels() -> None:
    labels: dict[str, list[Path]] = {}
    references: set[str] = set()
    for path in manuscript_markdown():
        text = path.read_text(encoding="utf-8")
        # Attributes may include Pandoc classes such as .unnumbered.
        for label in re.findall(
            r"^#{1,6}\s+.*\{[^}\n]*#(sec:[a-z0-9_-]+)(?=[\s}])", text, re.MULTILINE
        ):
            labels.setdefault(label, []).append(path)
        references.update(re.findall(r"@(sec:[a-z0-9_-]+)", text))
    missing = references - labels.keys()
    duplicate = {label: paths for label, paths in labels.items() if len(paths) != 1}
    assert not missing, f"Section references have no target: {sorted(missing)}"
    assert not duplicate, f"Section labels have ambiguous targets: {duplicate}"


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




def _defined_bib_keys() -> set[str]:
    """Citation keys defined in references.bib."""
    return set(BIB_KEY_RE.findall(REFERENCES_BIB.read_text(encoding="utf-8")))


def _cited_keys() -> set[str]:
    """Citation keys used in prose: ``[@key]`` markers, excluding pandoc-crossref
    labels (``@fig:``/``@sec:``/``@eq:``/``@tbl:``), which are followed by a colon."""
    cited: set[str] = set()
    for path in manuscript_markdown():
        text = path.read_text(encoding="utf-8")
        for match in CITATION_KEY_RE.finditer(text):
            if text[match.end() : match.end() + 1] != ":":
                cited.add(match.group(1))
    return cited


def test_no_legacy_numeric_citations_remain() -> None:
    """Citations are bibkey-based ([@key]); no hardcoded numeric [N] markers."""
    numeric = re.compile(r"\[\d+(?:,\s*\d+)*\]")
    for path in manuscript_markdown():
        hits = numeric.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} still has hardcoded numeric citations: {hits}"


def test_reference_ids_defined_and_used() -> None:
    defined = _defined_bib_keys()
    cited = _cited_keys()
    undefined = sorted(cited - defined)
    unused = sorted(defined - cited)
    assert defined, "references.bib defines no entries"
    assert not undefined, f"Manuscript cites keys absent from references.bib: {undefined}"
    assert not unused, f"references.bib defines keys never cited: {unused}"
