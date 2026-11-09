#!/usr/bin/env python3
"""Exercise the installed wheel, console entrypoint and packaged SPEC outside source.

Run with an isolated wheel environment, from a repository-owned temporary cwd
(or CI's runner temporary directory), never from source import roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if Path.cwd().resolve() in {source, source / "src"}:
        raise ValueError("Run installed-wheel smoke from a separate temporary working directory.")
    import alphacogant
    from alphacogant.api import Client
    from alphacogant.bridge.gnn import model_from_gnn, model_to_gnn

    installed = Path(alphacogant.__file__).resolve()
    if installed.is_relative_to(source / "src") or not installed.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise ValueError("Smoke did not import the isolated environment's installed wheel.")
    expected = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    assert alphacogant.__version__ == version("alphacogant") == expected
    spec = files("alphacogant").joinpath("SPEC.md").read_bytes()
    spec_sha = hashlib.sha256(spec).hexdigest()
    assert spec_sha == hashlib.sha256((source / "src/alphacogant/SPEC.md").read_bytes()).hexdigest()
    model_from_gnn(model_to_gnn(alphacogant.default_model()))
    assert Client().call("diagnostics")["healthy"]
    console = shutil.which("alphacogant")
    if console is None or Path(console).parent.resolve() != (Path(sys.prefix) / "bin").resolve():
        raise ValueError("Console entrypoint does not belong to the isolated wheel environment.")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    def invoke(*arguments, request=None, status=0):
        child = subprocess.run(
            [console, *arguments], input=request,
            stdin=subprocess.DEVNULL if request is None else None,
            cwd=Path.cwd(), env=environment, capture_output=True, text=True, timeout=60,
        )
        assert child.returncode == status, (child.returncode, child.stderr)
        return child.stdout

    assert invoke("--version").strip() == f"alphacogant {expected}"
    doctor = json.loads(invoke("doctor", "--json"))
    assert doctor["ok"] and doctor["result"]["healthy"]
    request = json.dumps({"id": "installed", "method": "simulation.run", "params": {
        "horizon": 2, "policy": "hold", "seed": 7,
    }}) + "\n"
    response = json.loads(invoke("request", request=request))
    assert response["id"] == "installed" and response["ok"]
    assert response["result"]["summary"]["horizon"] == "2"
    assert response["result"]["trajectory"]["action_history"] == [5, 5]
    import numpy as np

    huge_source = model_to_gnn(alphacogant.default_model(
        C_R=np.full(3, 1e307), C_L=np.zeros(3)
    ))
    overflow = {"id": "overflow", "method": "simulation.run", "params": {
        "source": huge_source, "horizon": 32, "policy": "hold", "seed": 7,
    }}
    frames = invoke("request", request=json.dumps(overflow) + "\n" + request, status=2)
    failed, recovered = [json.loads(line) for line in frames.splitlines()]
    assert failed["id"] == "overflow" and not failed["ok"]
    assert failed["error"]["code"] == "invalid_request"
    assert recovered["id"] == "installed" and recovered["ok"]
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="boundary-") as directory:
        model_path = Path(directory) / "extreme.gnn.md"
        output_path = Path(directory) / "must-not-exist.json"
        model_path.write_text(huge_source)
        rejected = json.loads(invoke(
            "simulate", "--source", str(model_path), "--horizon", "32",
            "--policy", "hold", "--json", "-o", str(output_path), status=2,
        ))
        assert not rejected["ok"] and rejected["error"]["code"] == "invalid_request"
        assert not output_path.exists()
    # Manuscript sources are deliberately not fabricated inside the wheel.
    try:
        alphacogant.generate_variables(n=2)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Wheel token generation must require manuscript sources.")
    print(json.dumps({"version": expected, "installed_module": str(installed),
                      "console": console, "spec_sha256": spec_sha,
                      "diagnostics": "healthy", "jsonl_policy": "hold", "horizon": 2,
                      "overflow_rejected": True, "next_frame_recovered": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
