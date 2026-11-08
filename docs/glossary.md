# Glossary: model, policy and evidence terms

## Action value

Negative Expected Free Energy: `value = -G = pragmatic + epistemic` for a discrete
move. See [EFE and policy](efe_and_policy.md).

## Artifact manifest

`output/reports/artifact_manifest.json`: hashes, sizes and structural issues for
current generated artifacts, not proof of current producer execution. See
[figures and artifacts](figures_and_artifacts.md).

## Dirichlet concentration (sometimes called belief precision)

The synthetic belief-perturbation parameter, not softmax policy precision.
It changes both the center and dispersion of the draw cloud. See
[t-RSI statistics](trsi_statistics.md) for the exact law and boundary clipping.

## Break-even probability

The fraction of paired synthetic draws with `create_rate > decay_rate`, not
financial success probability or a t-RSI confidence level. See
[t-RSI statistics](trsi_statistics.md).

## COGANT

An upstream codebase-to-GNN project. AlphaCOGANT instead authors a reduced model,
maps supplied firm counts to priors and exports numerical arrays. See the
[bridge contract](cogant_bridge.md) and [pinned interoperability limits](upstream_compatibility.md).

## EFE

Expected Free Energy `G`, minimized by the controller. Its information gain targets
Theta only. See [EFE and policy](efe_and_policy.md).

## EWM

Economic World Model. The historical corporate concept is learned and
filtration-respecting; this local illustration has fixed A/B/C/D and cannot
authenticate input chronology. See [model contract](generative_model_contract.md).

## Figure registry

`output/figures/figure_registry.json`: manuscript labels, captions, filenames and
expected producer paths, including the graphical abstract. See
[figures and artifacts](figures_and_artifacts.md).

## Filtration

Information available at decision time. The local filter/score/predict order is
inspectable; timestamps, training cutoffs and observation availability remain the
caller's responsibility. See [source-bound traces](cogant_bridge.md#source-bound-traces).

## Policy precision

Positive finite `gamma` in `softmax(gamma * value)`. It is neither Dirichlet
concentration nor an economic shadow price. See [EFE and policy](efe_and_policy.md).

## t-RSI and the certificate predicate

Paired synthetic mean create-minus-decay divided by its Monte Carlo SE; its
magnitude depends on draw count. `certificate(value, delta)` only compares scalars.
Neither simulator enforces it as an action or model-update admission gate. See
[t-RSI statistics](trsi_statistics.md).
