# Interpret paired synthetic create-minus-decay diagnostics

The implementation spans `alphacogant.trsi.t_rsi` and
`alphacogant.stats.statistics`. This page owns the rates, sampling and reporting
contract, separate from [EFE action selection](efe_and_policy.md). These are
synthetic reduced-model diagnostics, not empirical financial certificates.

## Average rates over post-action beliefs, not the initial state

`create_rate` averages greedy pragmatic value minus passive-hold pragmatic value
over H post-transition beliefs b1…bH. The final action contributes; H=1 includes
the first funded and hold outcomes. `decay_rate` averages the nonnegative
Theta-refresh opportunity gap on those same greedy states. The two quantities
are not algebraically ordered, and neither is epistemic information gain.

## Standardize the paired margin, retaining covariance

```text
t_rsi = mean(create - decay) / (std(create - decay, ddof=1) / sqrt(n))
```

Each pair comes from the same perturbed belief and trajectory. Covariance belongs
in the denominator; pooling marginal SE squares assumes independence and is not
the implemented formula. Centered dispersion is used when paired subtraction is
exact; adaptive rational arithmetic handles inexact/overflowing subtraction and
rounded-zero results. This preserves paired variation and cancellation residuals
that raw floating subtraction can erase even without overflow.
A truly constant nonzero margin yields signed infinity; a zero mean yields zero.
One paired sample returns zero by convention. Empty or unequal arrays are rejected.
An unrepresentable nonzero, nonconstant statistic raises `ValueError`, not a false
zero-variance infinity or rounded zero.

## Treat the threshold as a scalar predicate, not an admission mechanism

`certificate(value, delta)` compares scalars only. It is not enforced by the
simulators, compares no candidate/incumbent, and supplies no held-out/readiness
or monotonic-EFE guarantee.

## Interpret Dirichlet perturbations as synthetic sensitivity

The implementation draws `Dirichlet(concentration * maximum(belief, 1e-9) + 1)`
per channel, with explicit NumPy generators. At the interior operating beliefs,
the binary-factor mean is `(concentration*p+1)/(concentration+2)`:
concentration changes the center as well as dispersion and is not policy precision.
Canonical manuscript settings live in `model/operating_points.py`; the Python generator uses n=2560,
intentionally beyond the bounded agent API's n≤2048 interface.
Draw count and horizon must be positive integers; concentration must be positive
and finite. Reusing a seed makes a specified recipe reproducible, not empirical.

Percentile intervals summarize the perturbation cloud, not calibrated corporate
confidence or uncertainty of t-RSI itself. The t statistic's magnitude depends on
the generated sample count. `break_even_profile` separately reports positive-margin
event mass and paired margin percentiles; a majority-positive cloud can have a
negative mean if its negative tail is larger.

## Report effect sizes with actual confidence levels and sample counts

Cohen's d uses `(n_a-1)`/`(n_b-1)` weighted pooled sample variance. Groups must be
finite vectors with at least two entries. Identical constants yield zero;
distinct constants yield signed infinity, not a claim of no effect. Tables and
structured summaries retain each interval's actual confidence level and sample
count, including mixed levels between regimes.

Methodological and historical sources are identified in the
[manuscript bibliography](manuscript/references.bib) and
[qualified historical inventory](whitepaper_concepts.md).
