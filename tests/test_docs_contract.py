from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
REQUIRED_DOCS = {
    "README.md",
    "AGENTS.md",
    "architecture.md",
    "active_inference_dictionary.md",
    "generative_model_contract.md",
    "cogant_bridge.md",
    "efe_and_policy.md",
    "trsi_statistics.md",
    "figures_and_artifacts.md",
    "rendering_and_validation.md",
    "testing_contract.md",
    "claim_boundaries.md",
    "api_reference.md",
    "extending_the_model.md",
    "glossary.md",
    "methods_and_artifacts.md",
    "whitepaper_concepts.md",
}


def test_docs_hub_is_modular_and_indexed() -> None:
    docs = {path.name for path in DOCS_DIR.glob("*.md")}
    assert REQUIRED_DOCS.issubset(docs)
    assert len(docs) >= 17

    index = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
    for doc in REQUIRED_DOCS - {"README.md", "AGENTS.md"}:
        assert f"({doc})" in index, f"{doc} is not linked from docs/README.md"


def test_docs_files_have_top_level_headings() -> None:
    for path in sorted(DOCS_DIR.glob("*.md")):
        first_nonblank = next(
            line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        assert first_nonblank.startswith("# "), f"{path.name} lacks an H1"


def test_docs_preserve_claim_boundaries() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOCS_DIR.glob("*.md"))
    prose_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in DOCS_DIR.glob("*.md")
        if path.name != "claim_boundaries.md"
    )
    assert "does not reproduce AlphaFund's proprietary" in text
    assert "not provide financial advice" in text
    assert "the same control problem" not in prose_text
    assert "every allocation is a gradient" not in prose_text
