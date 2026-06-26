# GNN Model: AlphaFund Economic World Model as an Active Inference Agent
# GNN Version: 1.0
# AlphaCOGANT — the self-improving corporation rendered as a partially-observed
# Active Inference generative model. The firm's five capital channels are hidden
# state factors; capital allocation is the control vector; Expected Free Energy is
# the marginal-return objective the portfolio optimizer maximizes; the t-RSI
# certificate is the EFE-improvement gate on each commit.

## GNNSection
ActInfPOMDP_AlphaFund

## GNNVersionAndFlags
GNN v1

## ModelName
AlphaFund Economic World Model (five-channel self-improving corporation)

## ModelAnnotation
A partially-observed Active Inference model of the AlphaFund whitepaper's
self-improving corporation. The corporation tuple Ξ_t is factorized into the five
capital channels of the whitepaper:

- Factor I (Investments): the trading-book production channel. The only channel
  whose one-cycle return is read directly off the broker ledger (pragmatic).
- Factor S (Sensors): filtration breadth F_t = σ(H_t). Data feeds that let the
  generative model condition on more of the world (epistemic).
- Factor U (Actuators): the execution surface — venues, routes, tradable universe
  (pragmatic, gates how forecasts are realized).
- Factor Θ (Parameters): the learned EWM — predictive quality vs. staleness. This
  IS the generative model's own A/B; refreshing it is alpha-creation, ageing it is
  alpha-decay (both epistemic and pragmatic).
- Factor Z (R&D): research-frontier position — how fast the firm improves the
  other four channels (epistemic, compounds the whole loop).

Each factor carries 2 capability levels {weak=0, strong=1}. The control vector
a_t is the capital-allocation decision: which channel receives the marginal
dollar this cycle. Funding a channel raises the probability it transitions to /
stays strong; an unfunded Θ ages back toward weak (the empirically-measured
alpha-decay). Observations are the firm's noisy sensor readings of its own state
(the channel histories H_t^k): realized log-equity reward and EWM predictive loss.

The Expected Free Energy G(π) the controller minimizes is exactly the
whitepaper's marginal-return vector read as a path integral: its pragmatic term
is expected log-equity growth (the J_t objective, the create-rate), its epistemic
term is expected information gain about Θ (the data-scaling / forecast-sharpening
that Sensors and R&D buy). t-RSI is the standardized distance between the
create-rate posterior and the decay-rate posterior — the thresholded EFE gate
that admits a candidate update only when create confidently exceeds decay.

## StateSpaceBlock
# Five channel factors (the corporation tuple Ξ_t), each 2 capability levels
sI[2,1,type=float]      # Investments capability {weak, strong}
sS[2,1,type=float]      # Sensors / filtration breadth {weak, strong}
sU[2,1,type=float]      # Actuators / execution surface {weak, strong}
sTheta[2,1,type=float]  # Parameters / EWM quality {stale, fresh}
sZ[2,1,type=float]      # R&D frontier position {weak, strong}

# Likelihood (A) and transition (B) per factor
A_R[3,2,2,2,type=float] # Reward obs (3 levels) from sI, sU, sTheta (pragmatic readout)
A_L[3,2,2,type=float]   # Predictive-loss obs (3 levels) from sS, sTheta (epistemic readout)
B_I[2,2,6,type=float]   # Investments transition under allocation action
B_S[2,2,6,type=float]   # Sensors transition under allocation action
B_U[2,2,6,type=float]   # Actuators transition under allocation action
B_Theta[2,2,6,type=float] # Parameters transition; decays toward stale if unfunded
B_Z[2,2,6,type=float]   # R&D transition under allocation action

# Preferences, priors, observations, policy, EFE
C_R[3,type=float]       # log-preference over reward observation (pragmatic target)
C_L[3,type=float]       # log-preference over predictive-loss observation (prefer low loss)
D_I[2,type=float]       # prior over Investments capability
D_S[2,type=float]       # prior over Sensors capability
D_U[2,type=float]       # prior over Actuators capability
D_Theta[2,type=float]   # prior over Parameters capability
D_Z[2,type=float]       # prior over R&D capability
o_R[3,1,type=int]       # realized log-equity reward, bucketed {low, mid, high}
o_L[3,1,type=int]       # EWM predictive loss, bucketed {high, mid, low}
pi[6,type=float]        # policy: allocate to {I, S, U, Theta, Z, hold}
u[1,type=int]           # committed capital-allocation action
G[pi,type=float]        # Expected Free Energy per policy (the marginal-return vector)
g_epistemic[pi,type=float] # epistemic component of G (Sensors+R&D information gain)
g_pragmatic[pi,type=float] # pragmatic component of G (expected log-equity = create-rate)
tRSI[1,type=float]      # standardized create-vs-decay distance (EFE improvement gate)

## Connections
D_I>sI
D_S>sS
D_U>sU
D_Theta>sTheta
D_Z>sZ
sI-A_R
sU-A_R
sTheta-A_R
sS-A_L
sTheta-A_L
A_R-o_R
A_L-o_L
C_R>G
C_L>G
sS>g_epistemic
sZ>g_epistemic
sTheta>g_epistemic
sI>g_pragmatic
sU>g_pragmatic
g_epistemic>G
g_pragmatic>G
G>pi
pi>u
B_I>u
B_S>u
B_U>u
B_Theta>u
B_Z>u
u>sI
u>sS
u>sU
u>sTheta
u>sZ
g_pragmatic>tRSI
sTheta>tRSI

## InitialParameterization
# A_R: P(reward level | sI, sU, sTheta). Reward is high only when the production
# channels (I, U) are strong AND the EWM (Theta) is fresh enough to forecast edge.
# Indexed [reward_level][sI][sU][sTheta]; columns sum to 1 over reward_level.
A_R={
  ( ( (0.80,0.55),(0.55,0.30) ), ( (0.55,0.30),(0.30,0.10) ) ),
  ( ( (0.15,0.30),(0.30,0.40) ), ( (0.30,0.40),(0.40,0.30) ) ),
  ( ( (0.05,0.15),(0.15,0.30) ), ( (0.15,0.30),(0.30,0.60) ) )
}

# A_L: P(predictive-loss level | sS, sTheta). Loss is low (good) only when Sensors
# are rich AND Parameters fresh. Indexed [loss_level][sS][sTheta].
A_L={
  ( (0.70,0.45),(0.45,0.15) ),
  ( (0.22,0.35),(0.35,0.30) ),
  ( (0.08,0.20),(0.20,0.55) )
}

# Transition matrices: action index 0..5 = {fund_I, fund_S, fund_U, fund_Theta,
# fund_Z, hold}. Each B_k[next][prev][action]. Funding channel k pushes it toward
# strong; not funding it leaves it (mostly) put — except Theta, which decays.
B_I={
  ( (0.60,0.05), (0.95,0.95), (0.95,0.95), (0.95,0.95), (0.95,0.95), (0.90,0.10) ),
  ( (0.40,0.95), (0.05,0.05), (0.05,0.05), (0.05,0.05), (0.05,0.05), (0.10,0.90) )
}
B_S={
  ( (0.90,0.90), (0.55,0.05), (0.90,0.90), (0.90,0.90), (0.90,0.90), (0.90,0.10) ),
  ( (0.10,0.10), (0.45,0.95), (0.10,0.10), (0.10,0.10), (0.10,0.10), (0.10,0.90) )
}
B_U={
  ( (0.90,0.90), (0.90,0.90), (0.55,0.05), (0.90,0.90), (0.90,0.90), (0.90,0.10) ),
  ( (0.10,0.10), (0.10,0.10), (0.45,0.95), (0.10,0.10), (0.10,0.10), (0.10,0.90) )
}
# Theta decays: under any non-Theta action the fresh->stale leak is large
# (the empirically small-but-nonzero alpha-decay); funding Theta refreshes it.
B_Theta={
  ( (0.85,0.35), (0.85,0.35), (0.85,0.35), (0.50,0.02), (0.85,0.35), (0.85,0.40) ),
  ( (0.15,0.65), (0.15,0.65), (0.15,0.65), (0.50,0.98), (0.15,0.65), (0.15,0.60) )
}
B_Z={
  ( (0.90,0.90), (0.90,0.90), (0.90,0.90), (0.90,0.90), (0.55,0.05), (0.90,0.10) ),
  ( (0.10,0.10), (0.10,0.10), (0.10,0.10), (0.10,0.10), (0.45,0.95), (0.10,0.90) )
}

# Preferences (log scale, unnormalized). Strongly prefer high reward and low loss.
C_R={(-2.0, 0.0, 3.0)}
C_L={(-2.0, 0.0, 2.0)}

# Priors: the firm starts at the whitepaper's current operating point — small AUM,
# adequate sensors, fresh-ish EWM, active R&D.
D_I={(0.6, 0.4)}
D_S={(0.5, 0.5)}
D_U={(0.6, 0.4)}
D_Theta={(0.4, 0.6)}
D_Z={(0.5, 0.5)}

## Equations
# Generative model (filtration-respecting; conditions only on F_t = sigma(H_t)):
#   P(o_R, o_L, s_{t+1} | s_t, a_t) = A_R(o_R | sI,sU,sTheta) * A_L(o_L | sS,sTheta)
#                                     * prod_k B_k(s^k_{t+1} | s^k_t, a_t)
# Variational inference recovers the posterior q(s_t) over the five channel factors
# from the observed reward/loss history (the channel histories H_t^k).
#
# Expected Free Energy of a policy pi (negated marginal return the controller maximizes):
#   G(pi) = -g_pragmatic(pi) - g_epistemic(pi)
#   g_pragmatic(pi) = E_q[ ln P(o | C) ]            # expected log-preference = expected log-equity (create-rate)
#   g_epistemic(pi) = E_q[ D_KL[ q(Theta|o,pi) || q(Theta|pi) ] ]  # info gain on the EWM (Sensors+R&D)
# The marginal-return vector g_t = -dG/da_t equates the risk-adjusted shadow price
# of capital across funded channels at the optimum (equimarginal identity).
#
# t-RSI (improvement signal-to-noise ratio):
#   t-RSI = (mean[create-rate] - mean[decay-rate]) / sqrt(SE^2_create + SE^2_decay)
# create-rate posterior <- g_pragmatic bootstrap over channel-row fits
# decay-rate posterior  <- Theta fresh->stale leak (B_Theta) under non-funding
# Certificate of monotone improvement: admit a candidate Theta update iff
#   t-RSI >= delta (Sharpe-margin) on every active channel.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=Finite

## ActInfOntologyAnnotation
sI=HiddenStateFactor
sS=HiddenStateFactor
sU=HiddenStateFactor
sTheta=HiddenStateFactor
sZ=HiddenStateFactor
A_R=LikelihoodMatrix
A_L=LikelihoodMatrix
B_I=TransitionMatrix
B_S=TransitionMatrix
B_U=TransitionMatrix
B_Theta=TransitionMatrix
B_Z=TransitionMatrix
C_R=LogPreferenceVector
C_L=LogPreferenceVector
D_I=PriorOverHiddenStates
D_S=PriorOverHiddenStates
D_U=PriorOverHiddenStates
D_Theta=PriorOverHiddenStates
D_Z=PriorOverHiddenStates
o_R=Observation
o_L=Observation
pi=PolicyVector
u=Action
G=ExpectedFreeEnergy
g_epistemic=EpistemicValue
g_pragmatic=PragmaticValue
tRSI=ImprovementSignalToNoiseRatio

## ModelParameters
num_channel_factors: 5
num_levels_per_factor: 2
num_reward_levels: 3
num_loss_levels: 3
num_actions: 6
planning_horizon: 12

## Footer
AlphaFund Economic World Model v1 — GNN representation produced for AlphaCOGANT.
The five whitepaper channels (I, S, U, Theta, Z) are the hidden-state factors;
capital allocation is the control; Expected Free Energy is the portfolio
optimizer's marginal-return objective; t-RSI is the EFE-improvement certificate.
Reduced 2-level encoding for legibility; the manuscript carries the full continuous
marginal-return formalism.

## Signature
AlphaCOGANT / Active Inference Institute — illustrative model, not financial advice.
