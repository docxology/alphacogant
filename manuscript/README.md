# AlphaCOGANT manuscript

Writes out the AlphaCOGANT concept in full. Section order:

| File | Section |
| --- | --- |
| `00_abstract.md` | Abstract |
| `01_introduction.md` | RSI as economic control; the thesis; why route through AI + GNN |
| `02_active_inference_mapping.md` | The AlphaFund ↔ Active Inference dictionary |
| `03_gnn_via_cogant.md` | Technical/computational realization: GNN via COGANT |
| `04_generative_model_inference.md` | Inference under the firm filtration |
| `05_epistemic_and_pragmatic_value.md` | EFE decomposition; t-RSI as the certificate |
| `06_integrity_and_functionality.md` | What AlphaCOGANT brings: integrity properties |
| `07_conclusion.md` | Conclusion |
| `08_formalisms.md` | Numbered formalisms: AlphaFund definitions as Active Inference objects |
| `09_limitations.md` | Limitations, sensitivity analysis, and future directions |
| `99_references.md` | References |

## Token discipline

Every numeric cited in prose is a `{{TOKEN}}` produced by
`src/alphacogant/manuscript_variables.py::generate_variables` and written to
`output/manuscript_variables.json` by `scripts/z_generate_manuscript_variables.py`.
A token used in prose but absent from the generator is a build failure (no orphans).

Validate tokens locally:

```bash
uv run --no-project python scripts/z_generate_manuscript_variables.py --check
```
