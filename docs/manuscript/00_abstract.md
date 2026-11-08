# Abstract: an executable corporate analogy, not financial evidence {#sec:abstract .unnumbered}

The historical AlphaFund account frames recursive self-improvement (RSI) as a
portfolio optimization problem [@westenhaver2026rsi]. Its full primary text was
unavailable for independent rechecking. In that account, realized economic gains
finance better prediction and deployment, and **t-RSI** summarizes a gap between
alpha-creation and alpha-decay rates. **AlphaCOGANT** constructs an illustrative
reduced **Active Inference** analogy
[@friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis],
without asserting equivalence of the economic objectives.

We encode an illustrative **Economic World Model (EWM)** in **Generalized
Notation Notation (GNN)** using a **COGANT-inspired** structure-to-prior mapping
and independently checked matrix interchange [@friedman2026gnn; @friedman2026cogant].
The five named capital channels —
Investments, Sensors, Actuators, Parameters, and R&D — become the hidden-state
factors of a partially-observed model; capital allocation becomes a discrete
control action; and the portfolio optimizer's marginal-return objective motivates
**Expected Free Energy (EFE)** minimization in the reduced controller. Its
pragmatic term is expected reward/loss preference, not measured equity growth;
its epistemic term is information gain about the hidden Theta capability.
The matrices remain fixed: capability inference does not learn model parameters.
Separately, t-RSI standardizes paired synthetic create-minus-decay samples.
The threshold predicate is demonstrated, but neither simulator enforces it as an
action or model-update gate, and it does not certify monotonic EFE improvement.

We give the technical and computational realization: a GNN model file for the
five-channel firm, a tested NumPy Active Inference engine that performs state
inference, computes the epistemic/pragmatic EFE split and the marginal-return
vector, and evaluates the t-RSI diagnostic. The executable contribution is
inspectable filter/score/predict ordering, bounded finite-model interchange,
deterministic replay, and source-owned numerical and figure generation
[@friedman2026gnn; @friedman2026cogant]. Filtering is a single factorized sweep,
not exact joint inference. Chronological input integrity remains the caller's
responsibility; GNN structure alone cannot authenticate observation availability.
This synthetic modeling demonstration is not financial advice or evidence about
AlphaFund's proprietary books.

```{=latex}
\clearpage
```

![Graphical abstract of the reduced model: caller-supplied reward/loss observations pass through one factorized filtering sweep, one-step EFE scoring/selection and state prediction. Value = -G = pragmatic preference + Theta-information gain. A separate prediction-only greedy/hold diagnostic uses {{BOOTSTRAP_N}} paired synthetic draws, concentration {{BOOTSTRAP_CONCENTRATION}} and horizon {{PLANNING_HORIZON}}: mean margin {{BREAK_EVEN_MARGIN_MEAN}}, paired SE {{PAIRED_MARGIN_SE}}, t-RSI {{HEADLINE_T_RSI}} and positive-margin fraction {{BREAK_EVEN_PROB}} at IMPROVING. The scalar threshold does not gate allocation; A/B/C/D stay fixed, chronology is caller-owned, and no financial calibration or coefficient learning is implied.](../output/figures/cover_art.png){#fig:graphicalabstract}

```{=latex}
\clearpage
```
