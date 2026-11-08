# t-RSI and statistics

The implementation spans `trsi/t_rsi.py` and `stats/statistics.py`. It is a
synthetic reduced-model diagnostic, not an empirical financial certificate.

## Rates and timeline

`create_rate` averages greedy pragmatic value minus passive-hold pragmatic value
over H post-transition beliefs b1…bH. The final action contributes; H=1 includes
the first funded and hold outcomes. `decay_rate` averages the nonnegative
Theta-refresh opportunity gap on those same greedy states. The two quantities
are not algebraically ordered, and neither is epistemic information gain.

## Paired standardization

```text
t_rsi = mean(create - decay) / (std(create - decay, ddof=1) / sqrt(n))
```

Each pair comes from the same perturbed belief and trajectory. Covariance belongs
in the denominator; pooling marginal SE squares assumes independence and is not
the implemented formula. Scaling prevents intermediate underflow/overflow from
manufacturing a constant margin. A truly constant nonzero margin yields signed
infinity; zero margin or fewer than two paired samples returns zero.

`certificate(value, delta)` compares scalars only. It is not enforced by the
simulators, compares no candidate/incumbent, and supplies no held-out/readiness
or monotonic-EFE guarantee.

## Synthetic uncertainty

The law is `Dirichlet(concentration * belief + 1)` per channel, with explicit
NumPy generators. For a binary factor its mean is `(concentration*p+1)/(concentration+2)`:
concentration changes the center as well as dispersion. Canonical manuscript
settings live in `model/operating_points.py`; the Python generator uses n=2560,
intentionally beyond the bounded agent API's n≤2048 interface.

Percentile intervals summarize the perturbation cloud, not calibrated corporate
confidence or uncertainty of t-RSI itself. The t statistic's magnitude depends on
the generated sample count. `break_even_profile` separately reports positive-margin
event mass and paired margin percentiles; a majority-positive cloud can have a
negative mean if its negative tail is larger.

## Effect sizes and reporting

Cohen's d uses `(n_a-1)`/`(n_b-1)` weighted pooled sample variance. Groups must be
finite vectors with at least two entries. Identical constants yield zero;
distinct constants yield signed infinity, not a claim of no effect. Tables and
structured summaries retain each interval's actual confidence level and sample
count, including mixed levels between regimes.

Methodological and historical sources are identified in the
[manuscript bibliography](manuscript/references.bib) and
[qualified historical inventory](whitepaper_concepts.md).
