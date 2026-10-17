"""Deterministic observation-conditioned traces bound to exact GNN source."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np

from alphacogant.bridge.gnn import model_from_gnn, model_parameters
from alphacogant.efe.free_energy import (
    _loss_distribution,
    _reward_distribution,
    efe_vector,
    predicted_belief,
)
from alphacogant.model.channels import ACTIONS, CHANNELS
from alphacogant.model.generative_model import belief_prior, infer_states

MAX_TRACE_STEPS = 10_000


def _belief_lists(belief: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {c: belief[c].tolist() for c in CHANNELS}


def run_inference(
    source: str,
    *,
    steps: int = 8,
    observations: Sequence[Sequence[int]] | None = None,
) -> dict[str, object]:
    """Filter current observations, score actions, then predict the next belief.

    Without supplied observations, condition on modal model predictions. This
    is a synthetic model rollout with one mean-field filtering sweep, not market
    evidence or exact joint filtering. Ties choose the lowest action index.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be a positive integer.")
    if not 1 <= steps <= MAX_TRACE_STEPS:
        raise ValueError(f"steps must be in [1, {MAX_TRACE_STEPS}].")
    if observations is not None:
        if len(observations) != steps:
            raise ValueError("observations length must equal steps.")
        for obs in observations:
            if (
                not isinstance(obs, (list, tuple))
                or len(obs) != 2
                or any(
                    isinstance(v, (bool, np.bool_))
                    or not isinstance(v, (int, np.integer))
                    or not 0 <= v < 3
                    for v in obs
                )
            ):
                raise ValueError("Each observation must be a pair of integer buckets in [0, 2].")
    model = model_from_gnn(source)
    current = belief_prior(model)
    parameters = model_parameters(model)
    digest = hashlib.sha256()
    for name, array in parameters.items():
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    records = []
    for t in range(steps):
        reward = _reward_distribution(model, current)
        loss = _loss_distribution(model, current)
        obs = (
            (int(np.argmax(reward)), int(np.argmax(loss)))
            if observations is None
            else (int(observations[t][0]), int(observations[t][1]))
        )
        posterior = infer_states(model, obs[0], obs[1], current)
        scores = efe_vector(model, posterior)
        action = min(range(len(ACTIONS)), key=lambda a: scores[a].total)
        prediction = predicted_belief(model, posterior, action)
        records.append(
            {
                "t": t,
                "prior": _belief_lists(current),
                "observation": list(obs),
                "posterior": _belief_lists(posterior),
                "action": action,
                "action_name": ACTIONS[action],
                "prediction": _belief_lists(prediction),
                "efe": [
                    {"pragmatic": r.pragmatic, "epistemic": r.epistemic, "total": r.total}
                    for r in scores
                ],
            }
        )
        current = prediction
    return {
        "schema_version": "1.0",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "parameters_sha256": digest.hexdigest(),
        "mode": "model_rollout" if observations is None else "observations",
        "filter": "single_mean_field_sweep",
        "policy_horizon": 1,
        "policy_objective": "negative_expected_log_preference_minus_theta_information_gain",
        "steps": int(steps),
        "trace": records,
    }


def verify_inference(source: str, trace: dict[str, object]) -> None:
    """Replay the source and require every stored field to match."""
    try:
        mode = trace["mode"]
        if mode not in {"observations", "model_rollout"}:
            raise ValueError("Unknown observation mode.")
        observations = (
            None
            if mode == "model_rollout"
            else [record["observation"] for record in trace["trace"]]
        )
        expected = run_inference(source, steps=trace["steps"], observations=observations)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Malformed inference trace.") from exc
    if json.dumps(trace, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("Inference trace differs from source or replayed computation.")
