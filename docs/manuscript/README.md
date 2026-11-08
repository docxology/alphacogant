# AlphaCOGANT manuscript

Writes out the AlphaCOGANT concept in full. Section order:

| File | Section |
| --- | --- |
| `00_abstract.md` | Abstract |
| `01_introduction.md` | RSI as economic control; the thesis; why route through AI + GNN |
| `02_active_inference_mapping.md` | The AlphaFund ↔ Active Inference dictionary |
| `03_gnn_via_cogant.md` | Technical/computational realization: GNN via COGANT |
| `04_generative_model_inference.md` | Inference under the firm filtration |
| `05_epistemic_and_pragmatic_value.md` | EFE decomposition and separate paired t-RSI diagnostic |
| `06_integrity_and_functionality.md` | What AlphaCOGANT brings: integrity properties |
| `07_conclusion.md` | Conclusion (and Discussion subsection) |
| `08_formalisms.md` | Numbered formalisms: AlphaFund definitions as Active Inference objects |
| `09_limitations.md` | Limitations, sensitivity analysis, and future directions |
| `99_references.md` | References |

## Token discipline

Model-result numerics are tokens from
`src/alphacogant/tokens/manuscript_variables.py::generate_variables`.
Undefined referenced tokens fail generation. Coverage does not validate arbitrary
literals, interpretations or historical external claims.

Figure references are registered by the same script in
`output/figures/figure_registry.json`, using labels parsed from `docs/manuscript/*.md`.
The registry inventories expected producers; it does not attest that they ran.

Validate tokens locally:

```bash
uv run --locked python scripts/z_generate_manuscript_variables.py --check
```

Generate figures and variables before `--check`. Checking compares saved outputs
without rewriting them. See [contained rendering](../rendering_and_validation.md)
for Pandoc/PDF prerequisites and actual visual/reference validation.
