# AlphaCOGANT documentation: choose a task

The [manuscript](manuscript/README.md) develops the conceptual argument and its
synthetic evidence. These notes own the implementation and regeneration contracts;
the tables below identify one detailed owner for each subject.

## Read the argument and its limits

1. [Manuscript reading order](manuscript/README.md): abstract and graphical abstract,
   conceptual mapping, authored GNN, filtering, value, diagnostics and limitations.
2. [Claim boundaries](claim_boundaries.md): supported claims and explicit omissions.
3. [Historical concept inventory](whitepaper_concepts.md): attributed AlphaFund
   concepts, not independently reproduced financial evidence.
4. [Operational dictionary](active_inference_dictionary.md) and
   [glossary](glossary.md): short definitions with links to their contract owners.

## Compute, interchange or integrate

| Task | Detailed contract owner |
| --- | --- |
| Locate code and follow data flow | [architecture.md](architecture.md) |
| Construct A/B/C/D and filter observations | [generative_model_contract.md](generative_model_contract.md) |
| Map firm counts to priors; import/export GNN; replay traces | [cogant_bridge.md](cogant_bridge.md) |
| Understand bounded interoperability with pinned upstreams | [upstream_compatibility.md](upstream_compatibility.md) |
| Rank discrete actions by negative EFE | [efe_and_policy.md](efe_and_policy.md) |
| Interpret paired synthetic rates, t-RSI and event mass | [trsi_statistics.md](trsi_statistics.md) |
| Find Python imports and entry points | [api_reference.md](api_reference.md) |
| Call Python/JSONL methods and CLI commands | [agent_api.md](agent_api.md) |
| Measure runtime without overstating performance | [performance.md](performance.md) |
| Design an extension without claiming it already exists | [extending_the_model.md](extending_the_model.md) |

## Regenerate and review a manuscript

Start with the [source-to-output map](methods_and_artifacts.md), then follow
[rendering and validation](rendering_and_validation.md) for the single regeneration
and contained-render recipe required for manuscript releases. Detailed owners are
[figures and artifacts](figures_and_artifacts.md) for registry/manifest semantics
and [testing contract](testing_contract.md) for test coverage.

Do not repair generated output instead of its source. Token coverage, replay and
hashes establish different forms of consistency; none authenticates input chronology,
financial calibration, arbitrary prose or current producer execution. Inspect actual
figures and rendered pages before publication. External-template rendering is optional.
