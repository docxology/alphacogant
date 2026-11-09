# Bound interoperability to reviewed GNN and COGANT profiles

Reviewed on 2026-10-16 against public sources. The base package does not depend
on GNN or COGANT; its NumPy engine, finite-profile parser, and CLI run standalone.

| Upstream | Reviewed revision | Contract used |
| --- | --- | --- |
| [GNN](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/tree/2b105fde2c0f8fcd50eec5a45ab22d9925c01383) | `2b105fde2c0f8fcd50eec5a45ab22d9925c01383` | GNN v1.1 section/declaration/connection syntax and parameter axes |
| [COGANT](https://github.com/ActiveInferenceInstitute/COGANT/tree/5ea2a440bed9cd447e1dd70faae6ce49f0dfee8b) | `5ea2a440bed9cd447e1dd70faae6ce49f0dfee8b` (0.7.0) | Complete authoritative `gnn-matrices` fences and matrix-driven inference |

The GNN reference is the current [normative syntax specification](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/2b105fde2c0f8fcd50eec5a45ab22d9925c01383/docs/gnn/gnn_syntax.md).
The [COGANT inference guide](https://github.com/ActiveInferenceInstitute/COGANT/blob/5ea2a440bed9cd447e1dd70faae6ce49f0dfee8b/cogant/docs/api/inference.md)
and [finite-model guide](https://github.com/ActiveInferenceInstitute/COGANT/blob/5ea2a440bed9cd447e1dd70faae6ce49f0dfee8b/cogant/docs/api/finite_models.md)
separate authored declarations from evidence-bearing repository extraction.
AlphaCOGANT follows that distinction: successful compilation or execution
establishes behavior of this reduced model, not behavior of a real company.

## Check coefficients, joint axes and source-bound replay

The standard suite checks every authored GNN coefficient against the NumPy
engine, complete numeric round trips with custom coefficients, and an independent
enumeration of every joint A/B/C/D entry. It decodes the emitted fence independently
to check wire precision and action-axis ordering. Trace tests check timing,
source binding, deterministic replay, and tamper rejection.

Run the optional real COGANT parser and inference test with the reviewed revision:

```bash
uv sync --locked --extra dev
# A source checkout is enough; do not initialize upstream evaluation fixtures.
git clone --no-recurse-submodules https://github.com/ActiveInferenceInstitute/COGANT.git output/upstream/COGANT
git -C output/upstream/COGANT checkout --detach 5ea2a440bed9cd447e1dd70faae6ce49f0dfee8b
uv run --with './output/upstream/COGANT/cogant' pytest tests/test_cogant_integration.py -q
```

CI checks out the pinned source without submodules and runs this check separately
using the development interpreter discovered from `.python-version`.
The workflow verifies that actual runtime against the pin. Direct `git+`
package installation recursively fetches unrelated
upstream evaluation fixtures; a source-only checkout avoids that dependency and
its unrelated transfer/integrity failures. The base suite explicitly skips it
when COGANT is absent. The test compares every parsed matrix coefficient and
executes a three-observation trace in COGANT's actual runtime, checking the first
full-joint posterior against independent Bayes arithmetic and the selected
transition against matrix multiplication. It does not equate
COGANT's full-joint filtering or risk-plus-ambiguity objective with AlphaCOGANT's
mean-field filtering and Theta-information objective.

## Do not infer full-language support or equal policies from interchange

AlphaCOGANT implements a bounded finite GNN profile, not the complete GNN
language or its 25-step pipeline. Upstream schema checks do not prove scientific
validity. The joint export is a complete 32-state, nine-observation,
six-action model; it assumes the authored conditional independence structure.
COGANT extraction, finite-spec JSON compilation, reverse code synthesis, LLM
analysis, and alternate rendering backends are upstream capabilities, not claims
about the local bridge. Dependency updates require rerunning the integration
checks and updating this pinned record.
