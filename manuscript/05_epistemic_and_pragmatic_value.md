# Epistemic and pragmatic value, and t-RSI as the EFE certificate {#sec:value}

## Expected Free Energy is the marginal-return objective

At each cycle the controller scores a candidate policy $\pi$ (a capital
allocation) by its **Expected Free Energy**, which in Active Inference [3, 25] decomposes
into two terms with opposite signs of intent:

$$
G(\pi) = \underbrace{-\,\mathbb{E}_{q}\big[\ln P(o \mid C)\big]}_{\text{pragmatic cost}}
       \;-\; \underbrace{\mathbb{E}_{q}\big[D_{\mathrm{KL}}[\,q(s'\mid o,\pi)\,\Vert\,q(s'\mid \pi)\,]\big]}_{\text{epistemic value}} .
$$ {#eq:efe-objective}

The controller **minimizes** $G$; equivalently it maximizes value $= -G$. The two
parts answer two different questions about a marginal dollar:

- **Pragmatic value** — how much the action is expected to move outcomes toward
  preference. Here preference $C$ is high realized log-equity reward and low
  predictive loss, so pragmatic value is **expected log-equity growth**: AlphaFund's
  cumulative objective $J_t$, and the create-side of its marginal-return vector.
  Investments and Actuators ({{PRAGMATIC_CHANNELS}}) are the pragmatic channels —
  they realize edge now, read off the broker ledger.

- **Epistemic value** — how much the action is expected to *reduce uncertainty
  about the hidden state*, in particular the EWM parameters $\Theta$. This is the
  information gain a dollar buys, and it is exactly what AlphaFund's Sensors and
  R&D rows price: the data-scaling law (loss removed per decade of effective
  tokens) and the experiment-performance frontier (Sharpe gained per decade of
  experiments). Sensors and R&D ({{EPISTEMIC_CHANNELS}}) are the epistemic
  channels — they sharpen the model that prices all future outcomes [3, 25].

In AlphaFund's continuous formalism [1] the marginal-return vector is
$g_t = \partial J_t/\partial a_t$ and the optimum equates the same risk-adjusted
shadow price of capital — AlphaFund's equimarginal identity
$\hat g^k_t/\sigma^k_t = \lambda^*_{S,t}$. In AlphaCOGANT's reduced discrete
action space, `free_energy.marginal_return_vector` computes the corresponding
negative-EFE value for each of the {{NUM_ACTIONS}} admissible funding moves; its
argmax is the channel the portfolio optimizer funds this cycle. [@fig:heatmap]
shows the full vector as a heatmap over all cycles of the greedy trajectory,
making visible how the value landscape shifts as beliefs move from weak to
strong.

![Marginal-return vector (negative EFE = pragmatic + epistemic) across all six actions and all cycles of the greedy trajectory from the self-improving point. The starred action is the greedy selection. Computed by `simulation.simulate_trajectory` and `free_energy.marginal_return_vector`.](../output/figures/marginal_return_heatmap.png){#fig:heatmap}

[@fig:waterfall] decomposes the EFE itself — the objective the controller
minimizes — into its pragmatic and epistemic components for all six actions at
the IMPROVING operating point. The greedy action (lowest G, highest value) is
highlighted; the diamond markers show total G = -(pragmatic + epistemic) for
each action. The waterfall makes visible why the greedy policy funds
$\Theta$ when it is stale: the epistemic value of that funding is small in
absolute terms, but the pragmatic cost is the least negative of any action.

![Expected Free Energy decomposition per action at the IMPROVING operating point. G (◆) = -(pragmatic + epistemic); the greedy policy selects the lowest G. Computed by `free_energy.expected_free_energy`.](../output/figures/efe_waterfall.png){#fig:waterfall}

## Why the explore/exploit comparison stops being hard

AlphaFund's whole apparatus exists to make "a researcher hire, a data feed, a GPU,
a position in AAPL" comparable on one axis. The chronic difficulty is that a
trading position pays in dollars now while a data feed pays in *better future
prediction* — different units. Expected Free Energy resolves this because epistemic
value is denominated in the same nats-of-log-evidence that, through the likelihood,
convert into expected log-equity. A dollar on Sensors is worth the future pragmatic
value unlocked by the predictive loss it removes; a dollar on Investments is worth
the pragmatic value it realizes directly. The engine reports both parts for the
highest-scoring *funding* action: pragmatic {{FUNDED_PRAGMATIC}}, epistemic
{{FUNDED_EPISTEMIC}}, for {{FUNDED_CHANNEL}} — the channel the firm would back if it
committed capital this cycle. At the prior operating point the greedy policy in fact
*holds* (funding nothing scores above every allocation), which is exactly the
explore-versus-hold trade-off the next paragraph unpacks.

This also explains a structural feature of the self-improving corporation that
pure exploit accounting misses. When the EWM is **stale**, the epistemic value of
funding $\Theta$ or $S$ is large (there is much to learn), so the controller
explores; once the model is **fresh**, epistemic value falls and pragmatic value
dominates, so the controller exploits the now-accurate forecasts. The firm's
explore/exploit schedule is not a hand-tuned heuristic — it falls out of the EFE
decomposition as the belief over $\Theta$ tightens [23]. [@fig:trajectory] shows this
directly: from the self-improving point (stale $\Theta$), the greedy policy
funds $\Theta$ until it converges to fresh, then holds — the firm repairs its
most critical deficiency first, then switches to exploitation.

## t-RSI is the thresholded EFE-improvement certificate

AlphaFund's headline statistic, t-RSI, is the standardized distance between two
posteriors: alpha **created** per dollar (from the channel-row fits — the pragmatic
create-rate) and alpha **decayed** from the deployed book (from the
forecast-evaluation panel — the cost of *not* refreshing $\Theta$):

$$
\text{t-RSI}_{t:H}
  = \frac{\overline{\Delta\alpha}^{\,\text{create}}_{t:H} - \overline{\Delta\alpha}^{\,\text{decay}}_{t:H}}
         {\sqrt{\mathrm{SE}^2(\Delta\alpha^{\text{create}}_{t:H}) + \mathrm{SE}^2(\Delta\alpha^{\text{decay}}_{t:H})}} .
$$ {#eq:trsi-certificate}

In the AlphaCOGANT engine both rates are **path integrals over the planning
horizon**, matching AlphaFund's "path integral along the planned allocation path" [1],
and — critically — they are posteriors over **two genuinely different processes**,
so t-RSI is *not* constrained to be positive. `t_rsi.create_rate` is the
horizon-mean pragmatic value the greedy Expected-Free-Energy policy creates *over
passive holding*; `t_rsi.decay_rate` is the horizon-mean **residual** $\Theta$-
staleness erosion that *remains along the greedy trajectory* — a policy that keeps
refreshing $\Theta$ drives it toward zero, one that neglects $\Theta$ pays the full
freshness gap each cycle. Because the two share no algebraic term, create can
exceed decay (self-improvement) or fall below it (the firm bleeds). The engine's
`tests/test_t_rsi.py::test_comparator_is_not_green_by_construction` is the negative
control that proves this: it certifies that the self-improving operating point and
the coasting operating point order *oppositely*.

This honesty has a visible cost in the reduced two-level model. At the
self-improving operating point's *point estimate*, the active policy out-creates
its residual decay; but once `t_rsi.bootstrap_t_rsi` propagates belief uncertainty,
the headline reads a create-rate mean of {{CREATE_RATE_MEAN}}, a decay-rate mean of
{{DECAY_RATE_MEAN}}, and a headline t-RSI of {{HEADLINE_T_RSI}} standardized units.
The *raw* alpha gap is modest ({{CREATE_RATE_MEAN}} versus {{DECAY_RATE_MEAN}}), but
its *sign* is robust — create stays below decay across the n={{BOOTSTRAP_N}}
bootstrap. The standardized t-RSI inherits AlphaFund's pooled-standard-error
denominator, so its magnitude grows with the bootstrap count (it is a confidence
statistic, not an n-free effect size); read for sign and confidence it robustly
declines to certify net improvement. That is the correct behavior of an honest
instrument on a coarse encoding, not a defect: the two-level reduction lacks the
dynamic range to
*robustly* certify net self-improvement under self-knowledge uncertainty, and the
engine reports that rather than tuning the matrices until the number turns
favorable. A robustly positive headline (AlphaFund reports 9.61 on its proprietary
surfaces) is exactly what the full continuous marginal-return formalism is for, and
is deliberately out of scope here. The deliverable is the *machinery* — a
sign-discriminating certificate — not a manufactured headline.

[@fig:trsi] shows the two bootstrap posteriors directly: the create-rate
density centered at {{CREATE_RATE_MEAN}} and the residual-decay density centered at
{{DECAY_RATE_MEAN}}, their overlap making visible why the standardized gap reads
{{HEADLINE_T_RSI}} — a robust-sign separation (create below decay; a modest raw gap
whose standardized magnitude scales with the bootstrap count), not a manufactured win.

![Bootstrap posteriors of the create-rate (mean {{CREATE_RATE_MEAN}}) and residual decay-rate (mean {{DECAY_RATE_MEAN}}) at the self-improving operating point, using {{BOOTSTRAP_N}} deterministic Dirichlet perturbations; their standardized separation is the headline t-RSI {{HEADLINE_T_RSI}}, computed by `t_rsi.bootstrap_t_rsi`.](../output/figures/trsi_densities.png){#fig:trsi}

The sensitivity of this headline to the firm's belief precision is documented
in [@fig:sensitivity] and [@sec:limitations]. At low Dirichlet concentration the bootstrap
perturbations are wide and the standardized distance shrinks; at high
concentration they collapse and the distance inflates. The choice
$\alpha = 12$ is a modelling decision, not a tuned knob; the engine reports
the sensitivity rather than hiding it.

The **certificate of monotone improvement** is the thresholded form:
`t_rsi.certificate(value, delta)` admits a candidate self-improvement commit iff
t-RSI clears a Sharpe-margin $\delta$. This is the Active Inference admissibility
gate — a model update is accepted only when its expected free energy is reliably
lower than the incumbent's — and it is what makes compounding survive selection
rather than promoting drift on noise [1]. AlphaFund's claim that "the certificate gates
each commit at the prevailing operating point rather than relying on supermodularity
everywhere" is the standard Active Inference posture: value is evaluated locally,
per policy, per cycle, against the current belief, with no global guarantee assumed.

[@fig:certificate] shows where that gate is discriminating rather than
green-by-construction. It plots the point-estimate create-rate and decay-rate at
three operating points: at the self-improving point `create > decay` (the gate would
admit), while at the coasting point `create < decay` (the gate rejects). Because the
two rates order *oppositely* across regimes, the comparator cannot be
green-by-construction — a structurally `decay ≤ create` comparator could never
produce the coasting bar. This is exactly the property
`tests/test_t_rsi.py::test_comparator_is_not_green_by_construction` enforces.

Note the honest limitation, preserved from the headline above: once belief
uncertainty is bootstrapped, the *standardized* t-RSI is negative at **both** points
(the self-improving point reads {{HEADLINE_T_RSI}}; the coasting point reads a
large-magnitude {{COASTING_T_RSI}}, which is degenerate — the coasting greedy policy
is deterministically inert, so its create-variance collapses and inflates the
standardized distance). The reduced two-level encoding therefore does not *robustly*
certify net improvement under uncertainty; the sign-discrimination it does exhibit is
at the point-estimate level shown here, which is the load-bearing not-green-by-
construction evidence.

![Point-estimate create-rate (active policy) vs decay-rate (residual $\Theta$ staleness) at the prior, self-improving, and coasting operating points, from `t_rsi.create_rate` / `t_rsi.decay_rate`. The ordering flips — `create > decay` (ADMIT) at the self-improving point, `create < decay` (REJECT) at the coasting point — proving the comparator is not green-by-construction.](../output/figures/certificate_sign_flip.png){#fig:certificate}

## Standardization, not a hypothesis test

One subtlety the framing makes honest: t-RSI is a **standardized distance**, not a
hypothesis-test instrument. The create and decay posteriors are beliefs over two
different processes, not draws from one null. Reading t-RSI as "how many pooled
standard errors create sits above decay" is a calibrated effort-allocation signal,
not a p-value. The engine reflects this — it reports the separation and the pooled
standard error, and leaves the threshold $\delta$ as the firm's risk choice rather
than baking in a significance level [1, 17].
