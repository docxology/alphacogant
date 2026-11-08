# Epistemic and pragmatic value, and the separate t-RSI diagnostic {#sec:value}

## The implemented one-step objective

Active Inference motivates a preference/information decomposition
[@friston2015epistemic; @friston2017process]. In this reduced engine, an action
predicts the next product belief, and the implemented objective is

$$
G(a) = -\left(\mathbb E_q[C_R(o_R)+C_L(o_L)]
                 + I_q(\Theta';(o_R,o_L)\mid a)\right).
$$ {#eq:efe-objective}

The controller minimizes $G$, or equivalently maximizes value $=-G$.
`EFEResult.total == -(pragmatic + epistemic)`. The preference vectors are
finite authored log-preference scores, not observations of shareholders' equity.
The epistemic term is nonnegative information gain about the hidden Theta
capability only, not all state factors or learned matrix parameters. Sensors
influence the loss likelihood; R&D has no likelihood or cross-transition effect
in this encoding, so changing its belief cannot change the action scores.
The conceptual channel roles ({{PRAGMATIC_CHANNELS}} pragmatic,
{{EPISTEMIC_CHANNELS}} epistemic) must not be confused with demonstrated economic
returns [@westenhaver2026rsi].

`free_energy.marginal_return_vector` evaluates the {{NUM_ACTIONS}} discrete
negative-EFE scores. It is not the continuous derivative
$\partial J_t/\partial a_t$ or a solved risk-adjusted shadow-price condition.
A common unit in nats does not calibrate information gain into dollars
[@friston2010freeenergy; @dacosta2020synthesis].

![Negative-EFE action values over the {{PLANNING_HORIZON}}-cycle open-loop greedy trajectory from IMPROVING. Stars mark the actual one-step selections, not a multi-step optimal policy. Computed by `simulation.simulate_trajectory` and `free_energy.marginal_return_vector`.](../output/figures/marginal_return_heatmap.png){#fig:heatmap}

![Expected Free Energy decomposition per action at IMPROVING. Total G = -(pragmatic + epistemic); the greedy policy selects the lowest G. The two components are model scores, not financial measurements. Computed by `free_energy.expected_free_energy`.](../output/figures/efe_waterfall.png){#fig:waterfall}

At the neutral prior, the highest-scoring funding action is {{FUNDED_CHANNEL}},
with pragmatic {{FUNDED_PRAGMATIC}} and epistemic {{FUNDED_EPISTEMIC}}.
This excludes `hold` when identifying a funding channel; it does not claim that
funding beats the full action set. Along the IMPROVING greedy rollout, the first
action is {{TRAJ_FIRST_FUNDED}}, the last {{TRAJ_LAST_FUNDED}}, and the dominant
action {{TRAJ_DOMINANT_ACTION}}. A general exploration-to-exploitation story is
not a substitute for that observed sequence [@lattimore2020bandits].

## Paired synthetic create and decay rates

The names follow the motivating corporate analogy [@westenhaver2026rsi], but the
inputs here are synthetic belief perturbations, not fitted financial posteriors.
For a horizon of {{PLANNING_HORIZON}} actions, both rates evaluate the
post-transition states, excluding the shared initial state. `create_rate` is
the greedy trajectory's mean pragmatic value minus the passive-hold trajectory's
mean. `decay_rate` is the mean nonnegative pragmatic gap between each greedy
state and the same state with its Theta belief advanced by a refresh action.
It is a residual refresh opportunity, not an independently estimated market-decay
coefficient. The final action contributes to both horizon summaries.

Both samples in a pair use the same perturbed belief and greedy trajectory:

$$
\text{t-RSI}_{t:H}
  = \frac{\overline{\Delta\alpha^{\mathrm{create}}-
                              \Delta\alpha^{\mathrm{decay}}}}
         {\mathrm{SE}(\Delta\alpha^{\mathrm{create}}-
                      \Delta\alpha^{\mathrm{decay}})}.
$$ {#eq:trsi-certificate}

The paired denominator includes covariance; the sum of marginal SE squares
would instead assume independence. Scaled reductions avoid spurious zero or
infinite dispersion from tiny or large finite samples. An actually constant
nonzero margin has signed infinite t-RSI; an identical zero margin returns zero.

At the published settings ({{BOOTSTRAP_N}} draws, concentration
{{BOOTSTRAP_CONCENTRATION}}), create mean is {{CREATE_RATE_MEAN}}, decay mean
{{DECAY_RATE_MEAN}}, paired mean margin {{BREAK_EVEN_MARGIN_MEAN}}, paired SE
{{PAIRED_MARGIN_SE}}, and t-RSI {{HEADLINE_T_RSI}}. The generated paired
correlation is {{IMPROVING_RATE_CORRELATION}}. Separately,
{{BREAK_EVEN_PROB}} of draws satisfy `create > decay`; the paired percentile
interval is [{{MARGIN_CI_LOWER}}, {{MARGIN_CI_UPPER}}]. A majority-positive event
fraction can coexist with a negative mean when negative margins are larger.
Neither a mean's sign nor marginal intervals establish that every draw has the
same ordering.

![Synthetic Dirichlet-perturbation create and residual-decay densities at IMPROVING, using {{BOOTSTRAP_N}} draws. Means {{CREATE_RATE_MEAN}} and {{DECAY_RATE_MEAN}}, paired-margin SE {{PAIRED_MARGIN_SE}}, and t-RSI {{HEADLINE_T_RSI}} all use the same paired samples. Computed by `paired_bootstrap_samples` and `t_rsi`.](../output/figures/trsi_densities.png){#fig:trsi}

The perturbation law is $\mathrm{Dirichlet}(\alpha b_k+1)$ per binary channel.
Its mean is $(\alpha b_k+1)/(\alpha+2)$, not $b_k$: concentration changes
shrinkage toward the uniform belief as well as dispersion. The sample count
controls Monte Carlo precision, not the amount of observed corporate data.
Repeating draws can change the standardized magnitude without new evidence.
These are sensitivity diagnostics, not calibrated financial confidence or a
hypothesis test [@shannon1948].

## A scalar predicate, not enforced model admission

`t_rsi.certificate(value, delta)` evaluates `value >= delta`. It receives no
candidate/incumbent models, held-out observations, per-channel readiness, or
Fisher-information floor. Neither simulator calls it to admit actions or model
updates. The matrices remain fixed. The motivating corporate certificate is
therefore an analogy, not an implemented monotone-EFE-improvement guarantee.

![Point-estimate create and residual-decay rates at the prior, IMPROVING and COASTING beliefs. Positive/negative margin labels refer only to raw create-minus-decay, not the t-RSI threshold predicate or deployment admission. Computed by `create_rate` and `decay_rate`.](../output/figures/certificate_sign_flip.png){#fig:certificate}

The point-estimate comparator can order the rates either way; this is the
negative control against a structurally always-positive margin. The perturbed
COASTING statistic is {{COASTING_T_RSI}} and its positive-event fraction is
{{COASTING_BREAK_EVEN_PROB}}. Its magnitude depends on the paired margin's
variation and sample count, not the create variance alone. No sign or magnitude
is presented as evidence about AlphaFund's proprietary books.
