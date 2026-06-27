# AlphaCOGANT documentation hub

This directory is the modular technical companion to the manuscript. The
manuscript makes the argument; these notes describe the engine, artifacts, and
operational contracts used to regenerate and audit that argument.

## Core map

| File | Purpose |
| --- | --- |
| [architecture.md](architecture.md) | Package layout, ownership boundaries, and data flow. |
| [active_inference_dictionary.md](active_inference_dictionary.md) | AlphaFund terms mapped to Active Inference objects. |
| [generative_model_contract.md](generative_model_contract.md) | A/B/C/D matrix contracts and belief validation rules. |
| [cogant_bridge.md](cogant_bridge.md) | Firm-structure priors and GNN round-trip behavior. |
| [efe_and_policy.md](efe_and_policy.md) | Expected Free Energy, policy posterior, and discrete action values. |
| [trsi_statistics.md](trsi_statistics.md) | Create/decay rates, t-RSI, confidence intervals, and break-even profiles. |
| [figures_and_artifacts.md](figures_and_artifacts.md) | Figure scripts, registry, manifest, and provenance checks. |
| [rendering_and_validation.md](rendering_and_validation.md) | Project and template commands for regenerated PDFs. |
| [testing_contract.md](testing_contract.md) | No-mock tests, coverage expectations, and contract tests. |
| [claim_boundaries.md](claim_boundaries.md) | What the manuscript claims, and what it deliberately does not claim. |
| [api_reference.md](api_reference.md) | Stable public functions and which surface owns them. |
| [extending_the_model.md](extending_the_model.md) | Safe extension lanes for continuous state/action and learning dynamics. |
| [glossary.md](glossary.md) | Short definitions of local terms. |

## Related notes

- [methods_and_artifacts.md](methods_and_artifacts.md) records the source-to-output contract.
- [whitepaper_concepts.md](whitepaper_concepts.md) is the coverage checklist against the AlphaFund whitepaper.

## Ground rules

Do not edit `output/` to fix a documentation or manuscript defect. Change the
source file, regenerate figures and manuscript variables, then render through the
template pipeline.
