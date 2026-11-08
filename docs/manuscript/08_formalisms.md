# Numbered formalisms: implemented objects and conceptual analogies {#sec:formalisms}

The {{NUM_DEFINITIONS}} numbered definitions distinguish executable reduced-model
objects from the historical AlphaFund concepts that motivate them
[@westenhaver2026rsi]. They are not a proof that every corporate definition has an
implemented numerical counterpart. The {{NUM_FIGURES}} registered manuscript figures,
including the graphical abstract, combine model computations and labeled conceptual diagrams.

## Equity and cumulative return motivate, but do not calibrate, preference

**Definition D1 (Equity and preference).** Equity and solvency are external
corporate concepts. The engine has no assets/liabilities state or survival
constraint. `C_R` and `C_L` are finite authored log-preference scores;
negative-infinite preferences are rejected, not an implemented insolvency support
[@kelly1956].

**Definition D2 (Reward observation).** Reward is a discrete observation with
likelihood

$$
P(o_{R,t}\mid s_t^I,s_t^U,s_t^\Theta)
 = A_R(o_{R,t}\mid s_t^I,s_t^U,s_t^\Theta).
$$ {#eq:reward-likelihood}

It is not a measured log-equity ratio. `static_pragmatic_value` evaluates the
expected authored reward/loss preference at a supplied belief.

**Definition D3 (Cumulative objective analogy).** A corporate discounted return
objective motivates the representation, but the controller computes one-step
negative EFE. The create diagnostic instead contrasts mean pragmatic values of
greedy and hold trajectories over {{PLANNING_HORIZON}} actions. These are
separate operations, not an equality between economic $J_t$ and EFE
[@friston2010freeenergy; @dacosta2020synthesis].

## Product beliefs and discrete actions use authored factorized transitions

**Definition D4 (Product belief).** The engine retains channel beliefs in fixed
order `I,S,U,Theta,Z`:

$$
q(s_t)=\prod_{k\in\{I,S,U,\Theta,Z\}}q(s_t^k),\qquad q(s_t^k)\in\Delta^1.
$$ {#eq:state-factorization}

This is a chosen representation, not exact posterior independence after coupled
observations. `validate_belief_map` normalizes complete nonnegative finite maps.

**Definition D5 (Discrete action).** One of {{NUM_ACTIONS}} actions funds a single
channel or holds. No continuous dollar allocation or simultaneous multi-channel
portfolio solve is implemented.

**Definition D6 (Authored transition).** Under the reduced model,

$$
P(s_{t+1}\mid s_t,a_t)=\prod_k B_k(s_{t+1}^k\mid s_t^k,a_t).
$$ {#eq:transition-factorization}

Each `B[channel]` uses `[next,current,action]` axes. This is an assumed synthetic
law, not an identified true corporate transition [@kaelbling1998pomdp].

**Definition D7 (Fixed generative model).** `EconomicWorldModel` holds likelihoods,
transitions, preferences and priors. Theta is a hidden capability label, not
these numerical coefficients. No within-run coefficient learning is performed.

## Ordered observations do not authenticate external chronology

**Definition D8 (Observation history).** `run_inference` consumes supplied
reward/loss pairs in order, or explicitly labeled modal model observations.
Source hashes and deterministic replay check computation consistency, not whether
a supplied observation occurred in the external world.

**Definition D9 (Single factorized update).** `infer_states` averages likelihoods
over prior factors and updates individual channel vectors. Shared Theta couples
the observation evidence, so some cross-observation messages and posterior
correlations are discarded. Exact one-step marginals can differ even though
transitions factorize [@vandemeent2021ppl; @friston2017process].

**Definition D10 (Intended filtration).** A time-ordered predictor would use only
information available at the prediction time:

$$
\mathcal L_{\mathrm{EWM}}=
\sum_{\tau\in I_{\mathrm{eval}}}\ell\!\left(
\widehat P_\tau(o_{\tau+1}\mid\mathcal F_\tau,a_\tau),o_{\tau+1}\right).
$$ {#eq:ewm-predictive-loss}

This is a conceptual evaluation criterion, not a loss fitted here. The local
filter/score/predict order is inspectable, but input timestamps, corpus cutoffs
and retrospective contamination are not checked. GNN syntax does not make
future-derived input impossible [@westenhaver2026rsi].

**Definition D11 (Conditional independence).** Factorized transitions and coupled
likelihoods are distinct assumptions. The complete joint export preserves both;
the local filter additionally approximates its retained belief. Upstream joint
filtering need not agree with that approximation [@friedman2026gnn; @friedman2026cogant].

## One-step action scores and paired horizon rates answer different questions

**Definition D12 (Softmax policy).** `policy_posterior` computes

$$
q(a)=\operatorname{softmax}\!\left(\gamma[-G(a)]\right),\qquad\gamma>0.
$$ {#eq:policy-posterior}

Scalar precision is not a channel-specific risk vector or an equimarginal
shadow-price solve. A plotted softmax along a greedy rollout is not evidence that
the rollout sampled that distribution [@friston2015epistemic].

**Definition D13 (Finite action-value vector).**

$$
v_t[a]=-G(a)=\operatorname{pragmatic}(a)+\operatorname{epistemic}(a).
$$ {#eq:marginal-return-vector}

`marginal_return_vector` retains its historical API name but computes discrete
scores, not continuous derivatives. Its maximum includes `hold`; the separately
reported best funding channel {{FUNDED_CHANNEL}} excludes `hold`.

![Softmax action probabilities evaluated along the open-loop greedy trajectory. The trajectory's actual actions are greedy, not draws from these probabilities. Computed by `policy_posterior` and `simulate_trajectory`; no exploration-to-exploitation switch is assumed.](../output/figures/policy_posterior.png){#fig:policyposterior}

**Definition D14 (Horizon rate timeline).** Both rates evaluate H post-action
beliefs, so the final action outcome contributes. Creation is the mean greedy
pragmatic value minus mean passive-hold value. Decay is the mean nonnegative
Theta-refresh opportunity on those same greedy states. Neither is information
gain, and neither alone implements a monotone model-improvement proof.

**Definition D15 (Theta-information EFE).** The implemented decomposition is

$$
G(a)=-\mathbb E_q[C_R(o_R)+C_L(o_L)]
     -\mathbb E_o\!\left[D_{\mathrm{KL}}\!\left(
       q(\Theta'\mid o,a)\,\Vert\,q(\Theta'\mid a)\right)\right].
$$ {#eq:efe-decomposition}

The KL term concerns Theta only and is nonnegative. Expected preference includes
both reward and loss; it is not exclusively measured log-equity growth
[@friston2015epistemic; @friston2017process]. At the neutral prior, the best
funding action has pragmatic {{FUNDED_PRAGMATIC}} and epistemic
{{FUNDED_EPISTEMIC}}. Role labels {{EPISTEMIC_CHANNELS}} do not create absent
likelihood dependencies: Z has no effect on these scores.

![One-step pragmatic and Theta-information values for each action at the two illustrative operating beliefs. Shapes and coefficients come from `default_model`; values are computed by `expected_free_energy`. Differences between regimes do not demonstrate endogenous learning.](../output/figures/value_by_regime.png){#fig:valueregimes}

**Definition D16 (Paired standardized diagnostic).** For synthetic paired rates,

$$
\text{t-RSI}=\frac{\overline{c-d}}{\operatorname{std}(c-d,\mathrm{ddof}=1)/\sqrt n}.
$$ {#eq:trsi-formal}

A common perturbation feeds both rates; independent marginal SE pooling would
omit covariance. At the canonical settings the statistic is {{HEADLINE_T_RSI}},
paired mean margin {{BREAK_EVEN_MARGIN_MEAN}}, and positive-margin fraction
{{BREAK_EVEN_PROB}}. These quantities answer different questions. The magnitude
depends on generated draw count, not empirical data volume. It is not a
calibrated financial confidence level or a p-value.

**Definition D17 (Threshold predicate).**

$$
\operatorname{certificate}(x,\delta)\iff x\ge\delta.
$$ {#eq:admissibility-gate}

The function accepts scalars only. It does not compare a candidate with an
incumbent, verify held-out evidence or per-channel readiness, or gate either
simulator. The historical corporate admission concept is not implemented as a
deployment mechanism.

## Supermodularity, continuation and capital amplification remain conceptual

**Definition D18 (Supermodularity and continuation).** Economic cross-partials and
certified continuation bounds motivate possible extensions [@milgrom1990]. The
finite action vector has no implemented continuous mixed derivatives, and
likelihood coupling alone does not establish supermodularity. No continuation
bound is certified by this repository.

**Definition D19 (External capital).** The corporate decomposition

$$
K_{t+1}=K^{\mathrm{ext}}_{t+1}+K^{\mathrm{int}}_{t+1}
$$ {#eq:capital-decomposition}

is a conceptual reference [@westenhaver2026rsi; @coase1937], not an engine state
transition. Observation counts, filtration breadth and policy precision are not
endogenously increased by capital here. A source-owned representation helps make
such omissions explicit; it is not evidence that the omitted economics occurs.
