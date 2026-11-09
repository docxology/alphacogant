# Synthetic evidence limits and prerequisites for empirical extensions {#sec:limitations}

## Finite actions and product filtering omit corporate structure

Each of the {{NUM_CHANNELS}} channels has two capability levels, and the
controller chooses one of {{NUM_ACTIONS}} discrete actions. This is an
illustrative model, not a calibrated approximation error bound or a reproduction
of AlphaFund's proprietary books [@westenhaver2026rsi]. No equity, solvency,
continuous capital budget or risk-adjusted shadow-price solver is implemented.
A richer state space does not by itself establish better calibration or a
positive improvement statistic [@kaelbling1998pomdp].

The transition kernels factorize by channel, but coupled likelihoods induce
posterior dependence. `infer_states` keeps a single-sweep product approximation;
even individual factor marginals can differ from exact joint Bayes
[@vandemeent2021ppl; @friston2017process]. Complete joint matrix compilation is
exact under the authored independence assumptions; it does not make the local
factorized filter exact.

## Fixed matrices, no endogenous learning

Only beliefs move; $A,B,C,D$ remain fixed. Theta is a hidden capability label,
not the learned coefficients. Funding Theta changes its state distribution but
does not refit a world model. Z has an independent transition yet no likelihood
or cross-channel effect, so its belief cannot influence reward/loss or EFE.
The model does not implement supermodular transition effects or R&D learning
[@milgrom1990]. External capital amplification and per-channel readiness are
conceptual corporate ingredients, not executable features [@coase1937].

The inference loop handles supplied observations in order but cannot authenticate
their chronology. The separate trajectory simulator is open-loop prediction,
with no observed reward/loss filtering. Its {{PLANNING_HORIZON}} cycles do not
constitute multi-step optimal planning or a learned closed loop.

## Synthetic perturbations, not empirical confidence

The canonical diagnostic uses {{BOOTSTRAP_N}} seeded draws with concentration
{{BOOTSTRAP_CONCENTRATION}}. At the interior operating beliefs, the law
$\mathrm{Dirichlet}(\alpha b_k+1)$ smooths the center toward uniform and controls
dispersion; boundary entries are floored as described in [@sec:value]. Its
percentile intervals describe the synthetic draw cloud. They are not calibrated
confidence intervals for a real firm's returns. The t-RSI denominator is the
Monte Carlo SE of the paired margin, so its magnitude depends on draw count.
Neither a large magnitude nor a chosen threshold is an empirical significance
level [@shannon1948].

At IMPROVING, the reported mean margin is {{BREAK_EVEN_MARGIN_MEAN}}, t-RSI
{{HEADLINE_T_RSI}}, and positive-margin fraction {{BREAK_EVEN_PROB}}.
The paired central {{BOOTSTRAP_CONFIDENCE_PERCENT}}% interval is
[{{MARGIN_CI_LOWER}}, {{MARGIN_CI_UPPER}}]. The point-belief calculation and the
average over a smoothed perturbation distribution are different nonlinear
experiments. Variance alone cannot reverse the sign of a mean divided by a
nonnegative SE. No controlled ablation here identifies two-level coarseness as
the cause of a particular sign.

![Sensitivity diagnostics using the smaller sample recipe printed on the figure. Separate panels show dimensionless t-RSI against concentration and Theta freshness, and create/decay rate means against freshness. There is no t-RSI uncertainty band made from rate SEs. Concentration changes both center and dispersion. Computed by `sweep_concentration` and `sweep_theta_freshness`.](../output/figures/trsi_sensitivity.png){#fig:sensitivity}

The marginal create interval [{{CREATE_CI_LOWER}}, {{CREATE_CI_UPPER}}] and decay
interval [{{DECAY_CI_LOWER}}, {{DECAY_CI_UPPER}}] summarize different rate clouds;
whether either includes zero does not answer the paired-margin question.
Cohen's d values {{COHEN_D_CREATE}} and {{COHEN_D_DECAY}} compare the canonical
regime draws using pooled sample dispersion. They are descriptive effect sizes,
not an independent validation of financial utility.

![Regime comparison using the distinct sample recipe printed on the figure, not the canonical headline sample count. Left: central percentile intervals of synthetic create/decay rates. Right: one-step EFE decomposition of the selected action. Computed by `statistics.compare_regimes`.](../output/figures/regime_comparison.png){#fig:regimecomparison}

The canonical scatter uses the same per-regime seed and sample recipe as the
manuscript diagnostics. Positive margin lies below the diagonal because x is
create and y is decay. A mean's location and the fraction of points below the
line need not agree when tail magnitudes differ.

![Paired create-vs-decay scatter, {{BOOTSTRAP_N}} draws per regime at concentration {{BOOTSTRAP_CONCENTRATION}}. Each point is one synthetic belief perturbation; create > decay is below y=x. IMPROVING uses circles/X mean and COASTING triangles/P mean. Computed by `paired_bootstrap_samples`.](../output/figures/create_vs_decay_scatter.png){#fig:scatter}

Freshness profiles use a smaller, explicitly printed per-grid recipe and
seed offsets. They are not the identical canonical sample set. Event mass and
margin percentiles are separate diagnostics from a standardized mean.

![Synthetic positive-margin fraction and paired-margin central percentile interval across Theta freshness. The per-grid sample count, seed policy, concentration and horizon are printed on the figure. Computed by `statistics.break_even_profile`.](../output/figures/break_even_probability.png){#fig:breakeven}

## Prediction-only trajectories do not demonstrate observed learning

The IMPROVING greedy trajectory's first action is {{TRAJ_FIRST_FUNDED}}, last
{{TRAJ_LAST_FUNDED}}, and dominant action {{TRAJ_DOMINANT_ACTION}}. Its recorded
Theta-belief change is {{TRAJ_THETA_DELTA}}. The reported exploration ratio
{{TRAJ_EXPLORATION_RATIO}} counts S/Z actions over all cycles; it does not count
Theta repair and is not divided only by funded cycles. No eventual switch to
holding or exploitation is assumed.

![Open-loop greedy belief trajectories from IMPROVING and a separate fresh-Theta initial belief. Curves and action annotations are the actual predicted states and selected actions; no observations, matrix refitting or eventual-hold claim is implied. Computed by `simulation.simulate_trajectory`.](../output/figures/belief_trajectory.png){#fig:trajectory}

## Finite inputs do not guarantee representable diagnostic arithmetic

Paired floating subtraction can erase variation or a mean residual even without
overflow. The engine distinguishes genuine constant margins from rounded
coincidence using adaptive exact arithmetic. Unrepresentable nonconstant
statistics and cumulative trajectory totals fail explicitly rather than being
reported as numerical certificates. A truly constant nonzero paired margin has
zero sample SE and signed infinite standardized value by convention; that is not
evidence of corporate certainty or empirical improvement.

## Research directions and evidence prerequisites

Continuous actions, coupled transitions, endogenous parameter learning and
multi-step search could address specific representational omissions, but each
would require a new contract and independent verification. None guarantees a
positive t-RSI [@friston2017process]. Empirical calibration would require real,
chronologically admissible data, a stated sampling model and held-out evaluation.
Those inputs are not supplied here. Historical corporate claims are attributed
to the motivating whitepaper, whose current endpoint is unavailable as a full
primary document; they are not independently reproduced results.
