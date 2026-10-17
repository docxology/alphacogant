"""Real API requests, strict admission, seeded computations and JSONL subprocesses."""

import json
import subprocess
import sys

import numpy as np
import pytest

from alphacogant.api import Client, capabilities, handle_request, json_value
from alphacogant.bridge.gnn import model_to_gnn


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
    assert _cli("request", input="x" * 1000002).returncode == 2
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
