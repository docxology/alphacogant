# Limitations and future work {#sec:limitations}

## The two-level reduction

AlphaCOGANT's GNN model carries each channel at two capability levels
$\{\text{weak}, \text{strong}\}$. This is a deliberate legibility choice [@westenhaver2026rsi] — it makes
the factor graph readable, the inference exact, and the EFE decomposition
transparent — but it pays a measurable cost consistent with coarse-state Bayesian approximations [@vandemeent2021ppl; @kaelbling1998pomdp; @friston2017process]: the reduced model lacks the dynamic
range to *robustly* certify net self-improvement under belief uncertainty. The
headline t-RSI of {{HEADLINE_T_RSI}} standardised units is robustly negative in
*sign* at the self-improving operating point — a modest raw alpha gap whose
standardized magnitude scales with the bootstrap count n — not because the firm is not
self-improving (the point-estimate create-rate does exceed decay), but because
the two-level encoding's coarseness inflates the bootstrap variance enough to
overwhelm the signal. AlphaFund's published headline of 9.61 is exactly what the
full continuous marginal-return formalism is for; the reduced model delivers the
*machinery* — a sign-discriminating certificate — not a manufactured headline.

## No continuous capital allocation

The control vector is discrete: one of six actions (fund one of five channels, or
hold). AlphaFund's actual allocation is a continuous dollar-vector across
channels. The discrete reduction captures the explore/exploit logic and the
equimarginal identity in principle [@milgrom1990; @lattimore2020bandits], but it cannot represent a portfolio that
simultaneously funds Sensors and Investments at different intensities. A
continuous-action extension (e.g. via a Gumbel-softmax or a normalizing-flow
policy) would close this gap and is the natural next step.

## No learning dynamics

The model matrices $A, B, C, D$ are fixed. The EWM does not learn from
observations within a simulation run; only the *posterior over hidden state*
updates. In the real corporation [@westenhaver2026rsi], the EWM itself (the $\Theta$ factor's
parameters) is refit as new data arrives. A Bayesian model-learning extension —
where $B_\Theta$ is itself updated via a Dirichlet or Beta posterior over
transition probabilities — would make the self-forecasting loop endogenous [@vandemeent2021ppl; @kaelbling1998pomdp; @friston2017process].
The current model captures the *incentive* to refit (epistemic value of funding
$\Theta$) but not the *result* of refitting (sharper transition probabilities).

## No cross-channel coupling in the transition

The transition is fully factorised: $B_k(s^k_{t+1} \mid s^k_t, a_t)$ with no
cross-channel interaction. AlphaFund's Def 21 (supermodularity) is represented
in the *likelihood* (reward depends on $I, U, \Theta$ jointly) but not in the
*transition* (funding Sensors does not directly make Investments more
productive). A coupled transition — $B(s_{t+1} \mid s_t, a_t)$ rather than
\prod_k B_k$ would model supermodularity in the state dynamics, not just the
observation model [@milgrom1990; @kaelbling1998pomdp; @friston2017process]. The mean-field approximation is exact for the current factor
graph but would become variational under coupling [@vandemeent2021ppl; @kaelbling1998pomdp].

## No external capital amplification

The deployable-capital decomposition (Def 23, $K_{t+1} = K^{\text{ext}}_{t+1} +
K^{\text{int}}_{t+1}$) is described in the manuscript but not modelled in the
engine. External capital widens the filtration (more observations per cycle),
which sharpens the posterior and raises the policy precision $\gamma$. A model
that endogenises capital growth — where the number of observations per cycle is
a function of cumulative pragmatic value — would close this loop [@westenhaver2026rsi; @coase1937].

## Sensitivity to belief precision

[@fig:sensitivity] shows that the headline t-RSI is sensitive to the Dirichlet
concentration parameter that controls how tightly bootstrap perturbations hug the
operating belief. At low concentration ($\alpha \approx 2$) the perturbations are
wide and the standardised distance is small; at high concentration ($\alpha
\approx 80$) the perturbations collapse to a point mass and the distance
inflates. The choice $\alpha = 12$ (used throughout) is the firm's *belief
precision* — a modelling choice, not a tuned knob. A full Bayesian treatment
would place a hyperprior over $\alpha$ and marginalise; the current model reports
the sensitivity rather than hiding it.
This is a principled robustness check on the epistemic term [@shannon1948].

![t-RSI sensitivity to belief precision (left) and parameter freshness (right). Left: as Dirichlet concentration increases, perturbations tighten and the standardized distance changes. Right: as the Theta-freshness prior moves from stale to fresh, the create-rate and decay-rate means shift, and the t-RSI tracks their separation. Computed by `sensitivity.sweep_concentration` and `sensitivity.sweep_theta_freshness`.](../output/figures/trsi_sensitivity.png){#fig:sensitivity}

[@fig:regimecomparison] provides the key statistical summary: bootstrap 95%
confidence intervals for the create and decay rates at both operating points,
plus Cohen's d effect sizes for the between-regime differences. The CIs
overlap zero at the IMPROVING point (create CI [{{CREATE_CI_LOWER}},
{{CREATE_CI_UPPER}}]; decay CI [{{DECAY_CI_LOWER}}, {{DECAY_CI_UPPER}}]),
confirming the reduced model cannot *robustly* certify improvement — the
honest finding. Cohen's d for the create-rate difference between regimes is
{{COHEN_D_CREATE}}, and for the decay-rate difference is {{COHEN_D_DECAY}}.

![Regime comparison: bootstrap 95% confidence intervals for create and decay rates at the Improving and Coasting operating points (left), and EFE decomposition of the funded channel (right). CIs overlapping zero show the reduced model's inability to robustly certify improvement. Computed by `statistics.compare_regimes`.](../output/figures/regime_comparison.png){#fig:regimecomparison}

[@fig:scatter] shows the same data as a create-vs-decay scatter: each point is
one Dirichlet-perturbed belief, and the diagonal is the break-even line. The
Improving cloud (red) straddles the diagonal — some perturbations self-improve,
others bleed — while the Coasting cloud (green) sits consistently above it. The
standardized distance of each cloud from the diagonal is its t-RSI.

![Bootstrap create-rate vs decay-rate scatter at two operating points. Each point is one Dirichlet-perturbed belief; the diagonal is the break-even line (create = decay). ✗ marks the mean. Computed by `t_rsi.create_rate` / `t_rsi.decay_rate` with `n={{BOOTSTRAP_N}}` bootstrap perturbations.](../output/figures/create_vs_decay_scatter.png){#fig:scatter}

[@fig:breakeven] reports the paired event behind that scatter: the same
Dirichlet perturbation is used to compute create and decay, then the engine counts
whether `create_rate > decay_rate`. At the IMPROVING operating point,
{{BREAK_EVEN_PROB}} of paired bootstrap draws clear break-even, with mean paired
margin {{BREAK_EVEN_MARGIN_MEAN}} nats/cycle; the COASTING operating point clears
break-even in {{COASTING_BREAK_EVEN_PROB}} of draws. This event probability is not
a replacement for t-RSI, but it makes the reduced model's uncertainty easier to
read: the sign-discriminating comparator exists, yet the coarse two-level
encoding leaves substantial mass on both sides of zero.

![Break-even robustness across parameter-freshness beliefs. Left: paired bootstrap probability that create-rate exceeds decay-rate as the Theta-freshness prior changes. Right: the paired create-minus-decay margin with a 95% bootstrap interval. Computed by `statistics.break_even_profile`.](../output/figures/break_even_probability.png){#fig:breakeven}

## Trajectory analysis

[@fig:trajectory] shows the firm running for {{PLANNING_HORIZON}} cycles under
the greedy EFE policy from two starting points. From the self-improving point
(stale $\Theta$), the greedy policy funds $\Theta$ until it converges to fresh,
then holds — the firm repairs its most critical deficiency first. From a
fresh-$\Theta$ start (weak production channels), it briefly funds Sensors then
holds — with a sharp model, the marginal value of further exploration falls
below the cost. The funded action per cycle (annotated at the bottom of each
panel) is the EFE policy posterior's argmax; it is not a hand-tuned schedule.

![Belief trajectory under the greedy EFE policy. Left: from the self-improving operating point (stale $\Theta$), the policy funds $\Theta$ until it converges, then holds. Right: from a fresh-$\Theta$ start, it briefly funds Sensors then holds. Funded action per cycle is annotated at the bottom of each panel. Computed by `simulation.simulate_trajectory`.](../output/figures/belief_trajectory.png){#fig:trajectory}

[@fig:heatmap] decomposes the marginal-return vector (negative EFE) across all
six actions and all cycles of the greedy trajectory. The starred action is the
one selected each cycle. The value landscape shifts as beliefs move: early cycles
show high value on $\Theta$ (stale, much to learn); later cycles show convergence
toward `hold` as the model freshens and the marginal value of further funding
falls below the cost.



## Future directions

1. **Continuous state and action spaces.** Replace the two-level factors with
   continuous Beta-distributed capabilities and the discrete actions with a
   continuous allocation vector. This is the path to a robustly positive
   headline t-RSI.

2. **Endogenous model learning.** Make $B_\Theta$ a Dirichlet posterior that
   updates from observations, so the EWM genuinely learns within a run. This
   closes the self-forecasting loop.

3. **PyMDP / RxInfer cross-validation.** Export the GNN model file to a
   PyMDP or RxInfer.jl simulation and verify that the Active Inference
   computations (EFE, policy posterior, state inference) match the NumPy
   engine's output. This is the GNN pipeline's reason for existing.

4. **Multi-horizon planning.** The current engine plans one step ahead
   (greedy). A tree search or dynamic programming extension over the
   {{PLANNING_HORIZON}}-cycle horizon would compute the true optimal policy,
   not just the myopic one [@friston2017process].

5. **Empirical calibration.** Fit the model matrices to a real (or realistic
   synthetic) corporate trajectory and compare the engine's t-RSI to
   AlphaFund's published 9.61. This is out of scope (proprietary data) but
   the framework supports it.
