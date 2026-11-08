"""Real source-bound inference and subprocess CLI regression tests."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from alphacogant.bridge.gnn import model_to_gnn
from alphacogant.bridge.inference import run_inference, verify_inference
from alphacogant.efe.free_energy import efe_vector, predicted_belief
from alphacogant.model.generative_model import default_model, infer_states


def test_trace_orders_filter_score_predict_and_replays(model):
    source = model_to_gnn(model)
    trace = run_inference(source, steps=2, observations=[(2, 1), (0, 2)])
    verify_inference(source, json.loads(json.dumps(trace)))
    assert trace == run_inference(source, steps=2, observations=[(2, 1), (0, 2)])
    # The reference model has different array strides from the parsed model;
    # BLAS kernels can differ by one ulp while preserving the same coefficients.
    current = model.D
    for record in trace["trace"]:
        posterior = infer_states(model, *record["observation"], current)
        scores = efe_vector(model, posterior)
        assert record["action"] == min(range(6), key=lambda a: scores[a].total)
        for channel in current:
            np.testing.assert_allclose(
                record["prior"][channel], current[channel], atol=1e-15, rtol=0
            )
            np.testing.assert_allclose(
                record["posterior"][channel], posterior[channel], atol=1e-15, rtol=0
            )
        current = predicted_belief(model, posterior, record["action"])
        for channel in current:
            np.testing.assert_allclose(
                record["prediction"][channel], current[channel], atol=1e-15, rtol=0
            )


@pytest.mark.parametrize("mutation", ["posterior", "hash", "mode", "field", "missing"])
def test_replay_rejects_tampering(model, mutation):
    source = model_to_gnn(model)
    trace = copy.deepcopy(run_inference(source, steps=1))
    if mutation == "posterior":
        trace["trace"][0]["posterior"]["I"] = [0.5, 0.5]
    elif mutation == "hash":
        source += "\n# comment changes exact source bytes\n"
    elif mutation == "mode":
        trace["mode"] = "invalid"
    elif mutation == "field":
        trace["policy_horizon"] = 99
    else:
        del trace["trace"]
        trace["mode"] = "observations"
    with pytest.raises(ValueError):
        verify_inference(source, trace)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": True},
        {"steps": 1.5},
        {"steps": 10001},
        {"steps": 1, "observations": []},
        {"steps": 1, "observations": [(True, 1)]},
        {"steps": 1, "observations": [(3, 1)]},
        {"steps": 1, "observations": [1]},
    ],
)
def test_trace_admission(model, kwargs):
    with pytest.raises(ValueError):
        run_inference(model_to_gnn(model), **kwargs)


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "alphacogant", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_export_validate_infer_verify_and_exclusive_output(tmp_path):
    source = tmp_path / "model.gnn.md"
    result = _cli("export", "--cogant", "-o", source)
    assert result.returncode == 0, result.stderr
    original = source.read_bytes()
    assert _cli("export", "-o", source).returncode == 2
    assert source.read_bytes() == original
    assert _cli("validate", source).returncode == 0
    observations = tmp_path / "observations.json"
    observations.write_text("[[2,1],[0,2]]")
    trace = tmp_path / "trace.json"
    result = _cli("infer", source, "--observations", observations, "-o", trace)
    assert result.returncode == 0, result.stderr
    assert json.loads(trace.read_text())["steps"] == 2
    assert _cli("verify-trace", source, trace).returncode == 0
    assert _cli("export", "--source", source, "-o", source, "--overwrite").returncode == 0
    assert _cli("verify-trace", source, trace).returncode == 2
    assert _cli("export").stdout.startswith("## GNNSection")
    assert _cli("infer", source, "--steps", "1").returncode == 0
    assert _cli("validate", tmp_path / "missing").returncode == 2
    observations.write_text("broken json")
    assert _cli("infer", source, "--observations", observations).returncode == 2


def test_cli_in_process_runs_every_command_and_checks_errors(tmp_path, capsys):
    from alphacogant.__main__ import main

    source = tmp_path / "source.gnn.md"
    assert main(["export", "--cogant", "-o", str(source)]) == 0
    assert main(["validate", str(source)]) == 0
    assert "Valid" in capsys.readouterr().out
    trace = tmp_path / "trace.json"
    assert main(["infer", str(source), "--steps", "1", "-o", str(trace)]) == 0
    assert main(["verify-trace", str(source), str(trace)]) == 0
    assert "verified" in capsys.readouterr().out
    assert main(["export", "--source", str(source)]) == 0
    assert capsys.readouterr().out.startswith("## GNNSection")
    assert main(["export", "--source", str(source), "-o", str(source), "--overwrite"]) == 0
    assert main(["verify-trace", str(source), str(trace)]) == 2
    assert "differs" in capsys.readouterr().err
    observations = tmp_path / "obs.json"
    observations.write_text("[[1,2]]")
    assert main(["infer", str(source), "--observations", str(observations)]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "observations"
    assert main(["infer", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["steps"] == 8
    assert main(["export", "-o", str(source)]) == 2
    capsys.readouterr()
    observations.write_text("null")
    assert main(["infer", str(source), "--observations", str(observations)]) == 0
    capsys.readouterr()
    observations.write_text("broken")
    assert main(["infer", str(source), "--observations", str(observations)]) == 2
    assert "alphacogant:" in capsys.readouterr().err


def test_model_rollout_chooses_supported_joint_observation():
    reward = np.zeros((3, 2, 2, 2))
    loss = np.zeros((3, 2, 2))
    reward[0, :, :, 0] = 1.0
    reward[1, :, :, 1] = 1.0
    loss[1, :, 0] = 1.0
    loss[0, :, 1] = 1.0
    base = default_model()
    model = default_model(A_R=reward, A_L=loss, D={**base.D, "Theta": [0.5, 0.5]})
    source = model_to_gnn(model)
    # Both marginal modes select bucket0, but (0,0) has zero joint mass.
    with pytest.raises(ValueError, match="positive total probability"):
        run_inference(source, steps=1, observations=[(0, 0)])
    trace = run_inference(source, steps=2)
    assert trace["trace"][0]["observation"] == [0, 1]
    assert all(tuple(record["observation"]) in {(0, 1), (1, 0)} for record in trace["trace"])
    verify_inference(source, trace)


@pytest.mark.parametrize("preference", [1e308, -1e308])
def test_finite_preferences_cannot_publish_overflowed_action_scores(preference):
    source = model_to_gnn(default_model(C_R=[preference] * 3, C_L=[preference] * 3))
    with pytest.raises(ValueError, match="action scores must be finite"):
        run_inference(source, steps=1)


def test_large_representable_action_scores_still_replay():
    source = model_to_gnn(default_model(C_R=[1e150] * 3, C_L=[1e150] * 3))
    trace = run_inference(source, steps=1)
    assert all(
        np.isfinite(value)
        for record in trace["trace"]
        for score in record["efe"]
        for value in score.values()
    )
    verify_inference(source, json.loads(json.dumps(trace, allow_nan=False)))


@pytest.mark.parametrize("failure", ["integer", "score"])
def test_cli_numeric_failures_are_json_error_envelopes(tmp_path, failure):
    if failure == "integer":
        source = model_to_gnn(default_model()).replace(
            "D_I={(0.6, 0.4)}", "D_I={(" + "9" * 401 + ", 0)}"
        )
        args = ["validate"]
    else:
        source = model_to_gnn(default_model(C_R=[1e308] * 3, C_L=[1e308] * 3))
        args = ["infer"]
    path = tmp_path / "numeric-failure.gnn.md"
    path.write_text(source)
    result = _cli(*args, path, "--json")
    assert result.returncode == 2 and not result.stderr
    response = json.loads(result.stdout)
    assert not response["ok"]
    assert response["error"]["code"] == "invalid_request"
