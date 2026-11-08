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
| [rendering_and_validation.md](rendering_and_validation.md) | Contained standalone render, visual checks and optional template integration. |
| [testing_contract.md](testing_contract.md) | No-mock tests, coverage expectations, and contract tests. |
| [claim_boundaries.md](claim_boundaries.md) | What the manuscript claims, and what it deliberately does not claim. |
| [api_reference.md](api_reference.md) | Stable public functions and which surface owns them. |
| [extending_the_model.md](extending_the_model.md) | Safe extension lanes for continuous state/action and learning dynamics. |
| [glossary.md](glossary.md) | Short definitions of local terms. |

- [Agent API](agent_api.md) — versioned Python/JSONL requests, schemas, limits and CLI contracts.
- [Python 3.15 profiling](performance.md) — reproducible measurement and numerical compatibility.

## Related notes

- [methods_and_artifacts.md](methods_and_artifacts.md) records the source-to-output contract.
- [whitepaper_concepts.md](whitepaper_concepts.md) is the coverage checklist against the AlphaFund whitepaper.

## Ground rules

Do not edit generated output to fix a source defect. Regenerate figures and
manuscript variables, use the contained render recipe, then inspect actual output.
Structural checks and hashes do not prove empirical validity or semantic freshness.

- [Upstream compatibility](upstream_compatibility.md) — pinned GNN/COGANT contracts, checks, and limits.
