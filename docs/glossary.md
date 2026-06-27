# Glossary

## Action value

The negative Expected Free Energy value for one discrete funding move.

## Artifact manifest

`output/reports/artifact_manifest.json`, a hash-and-issue report for stable
generated artifacts.

## Belief precision

The Dirichlet concentration used when bootstrapping belief perturbations.

## Break-even probability

The paired bootstrap event mass where `create_rate > decay_rate`.

## COGANT

The codebase-to-GNN pattern. In AlphaCOGANT, the bridge maps firm structure to
priors and live model arrays back to a GNN-style summary.

## EFE

Expected Free Energy. AlphaCOGANT minimizes total EFE and ranks actions by
negative EFE.

## EWM

Economic World Model, AlphaFund's filtration-respecting world model.

## Figure registry

`output/figures/figure_registry.json`, a generated map from manuscript figure
references to filenames and producer scripts.

## Filtration

The information available at decision time. AlphaCOGANT's graph should not
route future observations into past beliefs.

## t-RSI

The standardized create-minus-decay separation used as a certificate signal.
