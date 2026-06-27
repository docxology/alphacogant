---
natbiboptions: numbers,square,sort&compress
biblio-style: unsrtnat
---

# Abstract {#sec:abstract}

The AlphaFund whitepaper reframes recursive self-improvement (RSI) as a portfolio
optimization problem: [@westenhaver2026rsi] a corporation recursively improves when realized economic
gains finance the next cycle of better prediction and deployment, and the firm's
standing is summarized by **t-RSI**, a standardized gap between alpha-creation and
alpha-decay rates. **AlphaCOGANT** observes that this construction is, term for
term, an **Active Inference** agent [@westenhaver2026rsi; @friston2010freeenergy; @friston2015epistemic; @dacosta2020synthesis] — and makes the correspondence executable.

We render AlphaFund's **Economic World Model (EWM)** as a generative model written
in **Generalized Notation Notation (GNN)**, produced by the **COGANT** [@friedman2026gnn; @friedman2026cogant]
codebase-to-GNN translation pattern. The firm's five capital channels —
Investments, Sensors, Actuators, Parameters, and R&D — become the hidden-state
factors of a partially-observed model; capital allocation becomes the control
vector; and the portfolio optimizer's marginal-return objective becomes
**Expected Free Energy (EFE)** minimization. The EFE decomposition supplies a
principled reading of AlphaFund's own categories: its **pragmatic value** is
expected log-equity growth (the alpha-creation rate, read off the broker ledger),
and its **epistemic value** is the information gain about the EWM that Sensors and
R&D purchase (the data-scaling and forecast-sharpening laws). t-RSI is recovered
as the standardized distance between the create-rate and decay-rate posteriors —
the thresholded EFE-improvement certificate that admits a self-improvement commit
only when creation confidently exceeds decay.

We give the technical and computational realization: a GNN model file for the
five-channel firm, a tested NumPy Active Inference engine that performs state
inference, computes the epistemic/pragmatic EFE split and the marginal-return
vector, and evaluates the t-RSI certificate. We argue that GNN-via-COGANT brings
two things AlphaFund's program needs and Active Inference already enforces:
**filtration integrity** (the model may condition only on information available
at decision time — the same "no-peeking" discipline that separates an EWM from a
language model) and **auditable capital allocation** (every admissible funding
move has a negative-EFE score under a single, legible objective). This is not
financial advice; it is a demonstration that this reduced
recursive-corporate-self-improvement model has a direct Active Inference
representation supported by source-owning methods and artifact checks [@westenhaver2026rsi; @friedman2026gnn; @friedman2026cogant].
