# Read the reduced-model argument and its evidence limits

The manuscript develops a conceptual corporate-RSI analogy, then separates the
implemented finite model from financial and theoretical claims it does not test.
Read the numbered files in this order; filenames and explicit cross-reference
labels are stable.

| File | Technical subject and evidence boundary |
| --- | --- |
| [00_abstract.md](00_abstract.md) | Abstract and registered graphical abstract: observation-conditioned controller versus separate synthetic diagnostic |
| [01_introduction.md](01_introduction.md) | Economic RSI motivates a bounded Active Inference analogy |
| [02_active_inference_mapping.md](02_active_inference_mapping.md) | Corporate concepts mapped to reduced objects, not objective identities |
| [03_gnn_via_cogant.md](03_gnn_via_cogant.md) | Authored GNN and bounded upstream matrix interchange |
| [04_generative_model_inference.md](04_generative_model_inference.md) | Single-sweep factorized filtering and caller-owned chronology |
| [05_epistemic_and_pragmatic_value.md](05_epistemic_and_pragmatic_value.md) | One-step Theta-information EFE versus separate paired diagnostics |
| [06_integrity_and_functionality.md](06_integrity_and_functionality.md) | Replay/token/hash consistency does not establish financial validity |
| [07_conclusion.md](07_conclusion.md) | Inspectability and prerequisites for stronger research evidence |
| [08_formalisms.md](08_formalisms.md) | Numbered implemented objects versus conceptual corporate definitions |
| [09_limitations.md](09_limitations.md) | Synthetic evidence, representational omissions and extension prerequisites |
| [99_references.md](99_references.md) | Cited methodological works and qualified historical attribution |

## Follow a question rather than reread every implementation detail

- **What is the argument?** Read the introduction, dictionary and conclusion.
- **What does the engine actually compute?** Read filtering, value and formalisms;
  use the [technical hub](../README.md) for implementation contract owners.
- **What can the evidence establish?** Read integrity, limitations and
  [claim boundaries](../claim_boundaries.md).
- **How are the pages regenerated?** Follow [source-to-output ownership](../methods_and_artifacts.md),
  [figure provenance](../figures_and_artifacts.md) and the manuscript-release
  [contained rendering recipe](../rendering_and_validation.md).

## Keep generated numerics and figure discovery in source

Model-result numerics come from
`alphacogant.tokens.manuscript_variables.generate_variables`. All Markdown in this
directory is scanned for referenced tokens and figure syntax, including this guide;
do not add placeholder figure embeds or undefined token examples here.

`cover_art.png` is embedded once at the end of the abstract as the graphical
abstract, with the label `fig:graphicalabstract`, and produced by
`scripts/figures/fig_cover_art.py`. Raw LaTeX clear-page blocks place it before the
introduction in PDF and are ignored by HTML. It is a registered figure with
provenance, not a decorative cover or an extra chapter.

Generate figures and variables before the non-writing `--check` step described in
[rendering and validation](../rendering_and_validation.md). Token coverage and the
figure registry do not attest arbitrary prose or prove that current producers ran.
