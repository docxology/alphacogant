"""Real API requests, strict admission, seeded computations and JSONL subprocesses."""

import json
import subprocess
import sys

import numpy as np
import pytest

from alphacogant.api import MAX_REQUEST_BYTES, Client, capabilities, handle_request, json_value
from alphacogant.bridge.gnn import MAX_GNN_BYTES, model_to_gnn
from alphacogant.model.generative_model import default_model


def test_discovery_and_all_engine_methods(model):
    client = Client()
    source = model_to_gnn(model)
    methods = capabilities()["methods"]
    methods["model.validate"]["params"]["properties"].clear()
    assert "source" in capabilities()["methods"]["model.validate"]["params"]["properties"]
    assert client.call("capabilities")["api_version"] == "1.0"
    assert client.call("diagnostics")["healthy"]
    assert client.call("model.validate", {"source": source})["valid"]
    assert client.call("model.export")["format"] == "gnn"
    assert client.call("model.export", {"source": source, "cogant": True})["format"] == "cogant"
    trace = client.call("inference.run", {"source": source, "observations": [[1, 2]]})
    assert trace["steps"] == 1
    assert client.call("inference.verify", {"source": source, "trace": trace})["verified"]
    simulation = client.call("simulation.run", {"horizon": 3, "policy": "stochastic", "seed": 19})
    assert simulation == client.call(
        "simulation.run", {"horizon": 3, "policy": "stochastic", "seed": 19}
    )
    assert len(simulation["trajectory"]["cycles"]) == 3
    stats = client.call("statistics.bootstrap", {"n": 4, "horizon": 2})
    assert stats == client.call("statistics.bootstrap", {"n": 4, "horizon": 2})
    result = handle_request({"id": "a", "method": "model.validate", "params": {"source": source}})
    assert result["ok"] and result["id"] == "a"
    assert json_value([np.array([np.inf, -np.inf, np.nan]), np.int64(2)]) == [
        ["Infinity", "-Infinity", "NaN"],
        2,
    ]
    json.dumps(simulation, allow_nan=False)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"method": []},
        {"method": "missing"},
        {"method": "capabilities", "extra": 1},
        {"method": "capabilities", "id": []},
        {"method": "capabilities", "id": True},
        {"method": "capabilities", "id": "x" * 129},
        {"method": "capabilities", "params": []},
        {"method": "capabilities", "params": None},
        {"method": "capabilities", "params": {"unknown": 1}},
        {"method": "model.validate"},
        {"method": "model.validate", "params": {"source": 1}},
        {"method": "model.validate", "params": {"source": "é" * 500001}},
        {"method": "model.export", "params": {"cogant": 1}},
        {"method": "statistics.bootstrap", "params": {"n": True}},
        {"method": "statistics.bootstrap", "params": {"n": 0}},
        {"method": "statistics.bootstrap", "params": {"n": 2049}},
        {"method": "statistics.bootstrap", "params": {"n": 2048, "horizon": 256}},
        {"method": "statistics.bootstrap", "params": {"concentration": float("nan")}},
        {"method": "statistics.bootstrap", "params": {"concentration": 10**500}},
        {"method": "statistics.bootstrap", "params": {"concentration": 0}},
        {"method": "simulation.run", "params": {"policy": "random"}},
        {"method": "inference.run", "params": {"source": "x", "observations": {}}},
        {"method": "inference.run", "params": {"source": "x", "observations": [[0, 0]] * 1001}},
        {"method": "inference.verify", "params": {"source": "x", "trace": {}}},
        {
            "method": "inference.verify",
            "params": {"source": "x", "trace": {"trace": [], "steps": True}},
        },
    ],
)
def test_structured_request_errors(payload):
    response = handle_request(payload)
    assert response["ok"] is False
    assert response["error"]["code"] in {"invalid_request", "method_not_found"}
    json.dumps(response, allow_nan=False)


def _cli(*args, input=None):
    return subprocess.run(
        [sys.executable, "-m", "alphacogant", *args],
        input=input,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_jsonl_mixed_requests_continue_and_report_exit_status():
    result = _cli(
        "request",
        input='{"id":1,"method":"capabilities"}\n'
        '{"id":2,"method":"missing"}\nbroken\n'
        '{"id":3,"method":"simulation.run","params":{"horizon":2}}\n',
    )
    assert result.returncode == 2 and not result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["ok"] for r in rows] == [True, False, False, True]
    assert rows[0]["id"] == 1 and rows[3]["id"] == 3
    assert _cli("request", input='{"method":"diagnostics"}\n').returncode == 0
    oversized = _cli("request", input="x" * (MAX_REQUEST_BYTES + 1) + "\n{}\n")
    assert oversized.returncode == 2 and not oversized.stderr
    assert len(oversized.stdout.splitlines()) == 1
    assert json.loads(oversized.stdout)["ok"] is False
    assert _cli("request", input='{}\n' * 1001).returncode == 2


def test_cli_machine_errors_and_new_commands(tmp_path):
    assert "alphacogant" in _cli("--version").stdout
    result = _cli("validate", str(tmp_path / "missing"), "--json")
    assert result.returncode == 2 and json.loads(result.stdout)["ok"] is False
    assert not result.stderr
    for args in [
        ("doctor", "--json"),
        ("capabilities",),
        ("simulate", "--horizon", "2"),
        ("bootstrap", "--n", "2", "--horizon", "2"),
        ("profile", "--n", "2", "--samples", "1", "--horizon", "1", "--no-memory"),
    ]:
        result = _cli(*args)
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)
    result = _cli("bootstrap", "--n", "0", "--json")
    assert result.returncode == 2 and not json.loads(result.stdout)["ok"]


def test_python_none_convenience_is_not_json_null_admission():
    assert Client().call("capabilities", None)["api_version"] == "1.0"
    assert handle_request({"method": "capabilities"})["ok"]
    response = handle_request({"method": "capabilities", "params": None})
    assert not response["ok"]
    assert response["error"]["code"] == "invalid_request"


def test_jsonl_parser_and_computation_errors_leave_next_frame_runnable(model):
    source = model_to_gnn(model)
    long_identifier = source.replace(
        "## InitialParameterization\n",
        "## InitialParameterization\n" + "X" * 100_000 + "\n",
    )
    huge_integer = source.replace("D_I={(0.6, 0.4)}", "D_I={(" + "9" * 401 + ", 0)}")
    parser_stack = source.replace("D_I={(0.6, 0.4)}", "D_I={(" + "+" * 10_000 + "0.6, 0.4)}")
    overflow = model_to_gnn(default_model(C_R=[1e308] * 3, C_L=[1e308] * 3))
    requests = [
        {"id": "identifier", "method": "model.validate", "params": {"source": long_identifier}},
        {"id": "integer", "method": "model.validate", "params": {"source": huge_integer}},
        {"id": "parser-stack", "method": "model.validate", "params": {"source": parser_stack}},
        {"id": "null", "method": "capabilities", "params": None},
        {"id": "overflow", "method": "inference.run", "params": {"source": overflow, "steps": 1}},
        {
            "id": "simulation-overflow", "method": "simulation.run",
            "params": {"source": overflow, "horizon": 1, "policy": "greedy"},
        },
        {
            "id": "export-overflow", "method": "model.export",
            "params": {"source": overflow, "cogant": True},
        },
    ]
    requests = [
        frame
        for request in requests
        for frame in (request, {"id": request["id"] + "-after", "method": "capabilities"})
    ]
    result = _cli("request", input="".join(json.dumps(row) + "\n" for row in requests))
    assert result.returncode == 2 and not result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["id"] for row in rows] == [row["id"] for row in requests]
    assert [row["ok"] for row in rows] == [False, True] * (len(requests) // 2)
    assert all(row["error"]["code"] == "invalid_request" for row in rows[::2])


def test_maximum_source_and_step_trace_replay_fits_jsonl_framing(model):
    limits = capabilities()["limits"]
    assert limits["request_bytes"] == MAX_REQUEST_BYTES == 32_000_000
    source = model_to_gnn(model)
    source += "#" + "x" * (MAX_GNN_BYTES - len(source.encode("utf-8")) - 1)
    assert len(source.encode("utf-8")) == limits["source_bytes"]
    request = {
        "id": "run",
        "method": "inference.run",
        "params": {"source": source, "steps": limits["inference_steps"]},
    }
    run_wire = json.dumps(request) + "\n"
    assert len(run_wire.encode("utf-8")) <= MAX_REQUEST_BYTES
    result = _cli("request", input=run_wire)
    assert result.returncode == 0 and not result.stderr
    response = json.loads(result.stdout)
    assert response["ok"]
    trace = response["result"]
    assert trace["steps"] == len(trace["trace"]) == 1000
    verify_wire = json.dumps({
        "id": "verify", "method": "inference.verify", "params": {"source": source, "trace": trace},
    }) + "\n"
    assert len(verify_wire.encode("utf-8")) <= MAX_REQUEST_BYTES
    result = _cli("request", input=verify_wire)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout)["result"] == {"verified": True}
