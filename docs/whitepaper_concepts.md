# AlphaFund whitepaper — concept inventory & AlphaCOGANT mapping rubric

Source: AlphaFund, *Recursive Self-Improvement is a Portfolio Optimization Problem*
(York Westenhaver, Massey Branscomb, Aidan Grant). This is the coverage checklist:
every concept below must be connected to an Active-Inference / GNN counterpart in the
AlphaCOGANT manuscript and, where it carries a number, an auto-injected `{{TOKEN}}`.

## Sections & core claims

1. **Introduction** — RSI is an economic problem under a survival constraint; every
   FLOP/bit costs money; intelligence = capacity to acquire, preserve, compound command
   over resources through accurate prediction; t-RSI = signal-to-noise of expected
   improvement vs posterior dispersion; AlphaFund 3-month t-RSI = 9.61; RSI cannot be
   guaranteed (uncertainty principle / gamma-ray-burst caveat).
2. **The Self-Improving Corporation** — perpetual succession / Ship of Theseus; the
   Corporate Loop as constrained stochastic optimal control over the production cycle.
3. **Firm Objective and Accounting Identity** — Def 1 Shareholders' equity
   K_t = Assets_t − Liabilities_t; Def 2 per-period reward R_τ = log(K_{t+1}/K_t)
   (Kelly time-average growth); survival constraint K_τ > 0; Def 3 cumulative objective
   J_t = E_{G,Ŵ_t}[ Σ R_τ | F_t ] (a DCF).
4. **The Corporation as a Bundle of Assets** — Def 4 Corporation tuple Ξ_t with state
   projection Π_state → (I,S,U,Θ,Z); Def 5 Action vector a_t (dollar change per channel);
   environment E_t.
5. **Coupled Dynamics** — Def 6 true corporate transition W(Ξ_{t+1},E_{t+1}|Ξ_t,E_t,a_t);
   small-firm approximation ∂E_{t+1}/∂a_t ≈ 0 (and Soros/Bank-of-England counter-example).
6. **The Self-Forecasting Loop** — predict (query Ŵ for ĝ_t) → optimize (convex inner
   program) → execute a*_t → realized (Ξ_{t+1},E_{t+1},R_t) becomes the next training row;
   filtration widening (Wayback/filings/alt-data); Figure 1.
7. **The Economic World Model** — Def 7 EWM Ŵ_t = filtration-respecting approx to W;
   LLMs are NOT EWMs (permutation-invariant L_LLM vs information-ordered L_EWM);
   Def 8 firm history H_t; Def 9 channel history H_t^k; Def 10 firm filtration F_t = σ(H_t);
   Def 11 channel-specific world model Ŵ_t^k.
8. **The Portfolio Optimizer** — Def 12 corporate optimization problem G* = argmax J_t s.t.
   constraints; Def 13 marginal-return vector g_t = ∂J_t/∂a_t; Def 14 per-channel marginal
   return (chain rule over the horizon); equimarginal identity ĝ^k_t/σ^k_t = λ*_{S,t};
   risk-neutral collapse κ_t→0.
9. **Experimental Evidence of t-RSI** — Def (Trsi Net) t-RSI = (Δα_create − Δα_decay)/
   sqrt(SE²_create + SE²_decay); the five channels as assets:
   - **Investments** Def 15 g^I_t = ∂/∂ΔI_t[ a^I_t·ΔI_t·μ̂_t − φ̂_t ] (instantaneous, broker ledger; ~$400M traded).
   - **Sensors** Def 16 local data-scaling slope; fitted α_Deff ≈ 0.156, R*=4.306, L_noise=0.042, A_Deff=3.266 (n=45, R²=0.762); Figure 3.
   - **Actuators** Def 17 local data-performance slopes; ∂AnnRet/∂log10 Deff=8.417, ∂Sharpe/∂log10 Deff=0.4696 (n=12, R²=0.80/0.95); Figures 4-5; tiers T_A/T_B.
   - **R&D** Def 18 experiments-performance slope; ∂Sharpe/∂log10(1+n)=0.3436 (R²=0.756,n=326), ∂AnnRet=2.135 (n=399); 929 experiments; selection-vs-structural caveat; Figures 6-7.
   - **Parameters** Def 19 g^Θ_t (Chinchilla joint surface); empirical alpha-decay λ_eff ~2-4×10⁻⁴/cycle, ~16mo Mark II/III, κ≈0; model-scaling sweep forthcoming.
   - **Continual Learning** Def 20 continual-learning intersection x_∩; α_1=0.0855, α_{K=3}=0.0280, x_∩=3.2×10¹⁴ DWT, L_∩=0.40 (n=45); Figure 8; prequential regime.
   - **Headline t-RSI** = 9.61; capacity sensitivity (4.59 at 10× AUM, 2.90 at 100×; crosses 0 between 10-20× worst case); ~$400k AUM, ~27× turnover; Figure 9.
10. **Toward the Differentiable Corporation** — every capital use is a row in a common
    optimization; **Three Structural Facts**: (a) pre-existing market with mandated price
    discovery, (b) principal value capture (∂Equity/∂Company dominated by own actions),
    (c) API-complete operational degrees of freedom; 15-bucket cross-industry exposure
    ranking (Quant Finance: LM 0.94, automation 0.76, API high).
11. **Channels Reinforce Each Other** — Def 21 cross-channel supermodularity ∂²J_t/∂a^j∂a^k ≥ 0
    (Milgrom-Roberts); Def 22 certified-commit continuation bound (probabilistic, not global).
12. **Drift Detection and Recovery** — three needs: fitted laws w/ SEs, drift detection,
    R&D refit; recovery rate > decay rate; Def 23 deployable-capital decomposition
    K_{t+1} = K^ext_{t+1} + K^int_{t+1}.
13. **The Bitter Lesson for Capital** — throughput grows with absorbed capital; external
    capital amplifies the loop while the marginal certificate clears; positive feedback.
14. **Completion Roadmap** — remaining gradients (salary, cost of capital, hardware, asset
    acquisition, AUM acquisition).
15. **Beyond Quant Trading** — three computational regimes (hand-factored chain rule →
    end-to-end neural EWM → differentiable policy/world-model, PILCO/Dreamer); the depth
    axis (Coasean verticalization); t-RSI practically computable here, ~undefined elsewhere.
16. **Conclusion** — differentiable corporation as a measurement architecture for
    compounding economic intelligence.

## Certificate of monotone improvement
Admit a candidate update iff held-out t-RSI clears Sharpe-margin δ and a Fisher-information
readiness floor ε_c on every active channel.

## The 23 numbered Definitions → AlphaCOGANT formalism targets
D1 equity↔preference target; D2 reward↔log-evidence; D3 J_t↔−E[EFE pragmatic]; D4 Ξ_t↔hidden
state; D5 a_t↔control; D6 W↔generative process; D7 Ŵ_t↔generative model; D8 H_t↔observation
sequence; D9 H_t^k↔per-factor evidence; D10 F_t↔measurability/no-peeking; D11 Ŵ_t^k↔factorized
model; D12 G*↔policy posterior; D13 g_t↔−∂G/∂a; D14 per-channel↔path integral; D15-19 channel
rows↔A/B sensitivities; D20 continual learning↔prequential refit; D21 supermodularity↔EFE
coupling; D22 continuation bound↔posterior-over-trajectory; D23 deployable capital↔external
evidence amplification.

## Numbers that MUST become auto-injected tokens (AlphaCOGANT model values, not AlphaFund's)
channel count, action count, levels/factor, planning horizon, B_Θ decay-leak probability,
prior/posterior beliefs, EFE pragmatic/epistemic per channel and per regime, marginal-return
argmax, t-RSI at prior/improving/coasting, create/decay means+SEs, certificate δ and verdicts,
number of definitions, number of figures, KL-nonnegativity check, determinism check.
