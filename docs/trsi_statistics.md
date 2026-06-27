# t-RSI and Statistics

The t-RSI stack spans `trsi/t_rsi.py` and `stats/statistics.py`, and implements the whitepaper's
standardized create-vs-decay certificate as
$\text{t-RSI} = (\Delta\alpha^\text{create} - \Delta\alpha^\text{decay}) / \sqrt{\text{SE}^2_\text{create} + \text{SE}^2_\text{decay}}$ [1].

## Rates

`create_rate()` is the horizon-mean pragmatic value created by the greedy EFE
policy over passive holding.

`decay_rate()` is the residual Theta-staleness erosion left along the greedy
trajectory. A policy that refreshes Theta can drive this down; a policy that
neglects Theta pays the freshness gap.

These are different processes. The comparator is therefore not green by
construction.

## t-RSI

`t_rsi(create_samples, decay_samples)` reports the standardized separation
between the two sample clouds:

```text
(mean(create) - mean(decay)) / sqrt(SE(create)^2 + SE(decay)^2)
```

It is a standardized distance, not a p-value.

## Bootstrap profiles

`bootstrap_t_rsi()` perturbs beliefs with a Dirichlet distribution controlled by
the concentration parameter and fixed random seeds. Manuscript tokens use the
canonical bootstrap constants from `model/operating_points.py`.

`break_even_profile()` uses paired perturbations: one perturbed belief is used to
compute both create and decay for a draw. This preserves the event structure of
`create_rate > decay_rate`.

## Reported statistics

`compute_regime_statistics()` and `compare_regimes()` provide confidence
intervals, effect sizes, funded-channel decomposition, and regime tables used by
figures and manuscript tokens.
