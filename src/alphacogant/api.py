"""Versioned, schema-described agent API shared by Python and JSONL transports.

Methods operate only on supplied content and finite built-in computations. They
cannot execute commands, write files, load plugins, or access network services.
"""

from __future__ import annotations

import dataclasses
import math
import platform
from collections.abc import Mapping
from copy import deepcopy
from importlib.metadata import version

import numpy as np

from alphacogant.bridge.gnn import (
    model_from_gnn,
    model_parameters,
    model_to_cogant_gnn,
    model_to_gnn,
)
from alphacogant.bridge.inference import run_inference, verify_inference
from alphacogant.model.generative_model import belief_prior, default_model
from alphacogant.profiling import profile_workload, source_fingerprint
from alphacogant.stats.simulation import simulate_trajectory, summarize_trajectory
from alphacogant.trsi.t_rsi import bootstrap_t_rsi

API_VERSION = "1.0"


def _integer(default: int, maximum: int, minimum: int = 1):
    return {"type": "integer", "minimum": minimum, "maximum": maximum, "default": default}


_SOURCE = {"type": "string", "maxLength": 1_000_000}
_SEED = _integer(7, 2**32 - 1, 0)
_METHODS = {
    "capabilities": ({}, [], "List API methods, schemas and execution limits."),
    "diagnostics": ({}, [], "Report runtime identity and exercise the installed engine."),
    "model.validate": ({"source": _SOURCE}, ["source"], "Validate finite GNN content."),
    "model.export": (
        {"source": _SOURCE, "cogant": {"type": "boolean", "default": False}},
        [],
        "Export complete coefficients, optionally with COGANT joint matrices.",
    ),
    "inference.run": (
        {
            "source": _SOURCE,
            "steps": _integer(8, 1000),
            "observations": {
                "type": ["array", "null"],
                "minItems": 1,
                "maxItems": 1000,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer", "minimum": 0, "maximum": 2},
                },
            },
        },
        ["source"],
        "Filter, score and predict a replayable trace.",
    ),
    "inference.verify": (
        {"source": _SOURCE, "trace": {"type": "object"}},
        ["source", "trace"],
        "Strictly replay a saved trace.",
    ),
    "simulation.run": (
        {
            "source": _SOURCE,
            "horizon": _integer(12, 1000),
            "seed": _SEED,
            "policy": {
                "type": "string",
                "enum": ["greedy", "hold", "fund_theta", "stochastic"],
                "default": "greedy",
            },
        },
        [],
        "Simulate an allocation trajectory; this does not filter observations.",
    ),
    "statistics.bootstrap": (
        {
            "source": _SOURCE,
            "n": _integer(32, 2048),
            "horizon": _integer(12, 256),
            "seed": _SEED,
            "concentration": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 1e9,
                "default": 12.0,
            },
        },
        [],
        "Compute paired create/decay bootstrap statistics.",
    ),
    "profile.run": (
        {
            "workload": {
                "type": "string",
                "enum": ["bootstrap", "inference", "efe"],
                "default": "bootstrap",
            },
            "samples": _integer(5, 15),
            "n": _integer(32, 2048),
            "horizon": _integer(12, 256),
            "seed": _SEED,
            "memory": {"type": "boolean", "default": True},
        },
        [],
        "Measure bounded seeded workloads with source and runtime provenance.",
    ),
}


def capabilities() -> dict[str, object]:
    from alphacogant import __version__

    return {
        "api_version": API_VERSION,
        "package_version": __version__,
        "methods": {
            name: {
                "description": description,
                "params": {
                    "type": "object",
                    "properties": deepcopy(properties),
                    "required": required.copy(),
                    "additionalProperties": False,
                },
            }
            for name, (properties, required, description) in _METHODS.items()
        },
        "limits": {
            "source_bytes": 1_000_000,
            "request_bytes": 1_000_000,
            "requests_per_stream": 1000,
            "bootstrap_n_times_horizon": 65_536,
        },
        "nonfinite_result_encoding": "Infinity, -Infinity and NaN are JSON strings",
    }


def json_value(value):
    """Encode numerical results without emitting nonstandard JSON numeric literals."""
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _admit(method: str, params: object) -> dict:
    if method not in _METHODS:
        raise LookupError(f"Unknown method: {method}.")
    properties, required, _ = _METHODS[method]
    if not isinstance(params, dict) or set(params) - set(properties):
        raise ValueError("params must be an object containing only declared properties.")
    if any(key not in params for key in required):
        raise ValueError(f"Required parameters: {', '.join(required)}.")
    admitted = {}
    for key, spec in properties.items():
        if key not in params and "default" not in spec:
            continue
        value = params.get(key, spec.get("default"))
        kind = spec["type"]
        if kind == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if valid:
                try:
                    valid = math.isfinite(value)
                except OverflowError:
                    valid = False
        elif kind == "boolean":
            valid = isinstance(value, bool)
        elif kind == "string":
            valid = isinstance(value, str)
        elif kind == "object":
            valid = isinstance(value, dict)
        else:
            valid = value is None or isinstance(value, list)
        if not valid:
            raise ValueError(f"{key} must have type {kind}.")
        if isinstance(kind, str) and kind in {"integer", "number"}:
            if value < spec.get("minimum", -math.inf) or value > spec.get("maximum", math.inf):
                raise ValueError(f"{key} is outside its declared range.")
            if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
                raise ValueError(f"{key} must be > {spec['exclusiveMinimum']}.")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} must be one of {spec['enum']}.")
        if isinstance(value, str) and len(value.encode("utf-8")) > spec.get("maxLength", 1_000_000):
            raise ValueError(f"{key} exceeds its byte limit.")
        if isinstance(value, list) and len(value) > spec.get("maxItems", 1000):
            raise ValueError(f"{key} exceeds its item limit.")
        admitted[key] = value
    if admitted.get("n", 1) * admitted.get("horizon", 1) > 65_536:
        raise ValueError("n * horizon must be <= 65536.")
    return admitted


class Client:
    """Synchronous in-process API; ``call`` returns data or raises admission errors."""

    def call(self, method: str, params: dict | None = None):
        p = _admit(method, {} if params is None else params)
        if method == "capabilities":
            return capabilities()
        if method == "profile.run":
            return profile_workload(**p)
        if method == "diagnostics":
            from alphacogant import __version__

            model_from_gnn(model_to_gnn(default_model()))
            installed = version("alphacogant")
            return {
                "healthy": installed == __version__,
                "package_version": __version__,
                "distribution_version": installed,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "source_sha256": source_fingerprint(),
                "model_round_trip": "passed",
            }
        if method == "inference.run":
            if (
                "observations" in p
                and p["observations"] is not None
                and "steps" not in (params or {})
            ):
                p["steps"] = len(p["observations"])
            return run_inference(**p)
        if method == "inference.verify":
            if not isinstance(p["trace"].get("trace"), list) or len(p["trace"]["trace"]) > 1000:
                raise ValueError("trace must contain at most 1000 records.")
            steps = p["trace"].get("steps")
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 1000:
                raise ValueError("trace steps must be an integer in [1, 1000].")
            verify_inference(**p)
            return {"verified": True}
        model = default_model() if "source" not in p else model_from_gnn(p.pop("source"))
        if method == "model.validate":
            return {
                "valid": True,
                "parameters": {name: list(a.shape) for name, a in model_parameters(model).items()},
            }
        if method == "model.export":
            render = model_to_cogant_gnn if p["cogant"] else model_to_gnn
            return {"source": render(model), "format": "cogant" if p["cogant"] else "gnn"}
        if method == "simulation.run":
            trajectory = simulate_trajectory(model, belief_prior(model), **p)
            return json_value(
                {"summary": summarize_trajectory(trajectory), "trajectory": trajectory}
            )
        return json_value(
            bootstrap_t_rsi(model, belief_prior(model), np.random.default_rng(p.pop("seed")), **p)
        )


def handle_request(request: object) -> dict[str, object]:
    """Return one bounded-request response; malformed inputs never invoke a method."""
    identity = None
    try:
        if not isinstance(request, dict) or set(request) - {"id", "method", "params"}:
            raise ValueError("request must contain only id, method and params.")
        identity = request.get("id")
        if not (
            identity is None
            or (isinstance(identity, (str, int)) and not isinstance(identity, bool))
        ):
            identity = None
            raise ValueError("id must be a string, integer or null.")
        if isinstance(identity, str) and len(identity) > 128:
            identity = None
            raise ValueError("id must have at most 128 characters.")
        method = request.get("method")
        if not isinstance(method, str):
            raise ValueError("method must be a string.")
        result = Client().call(method, request.get("params", {}))
        return {
            "schema_version": API_VERSION,
            "id": identity,
            "ok": True,
            "result": json_value(result),
        }
    except (ValueError, TypeError, LookupError, OSError, RecursionError) as exc:
        code = "method_not_found" if isinstance(exc, LookupError) else "invalid_request"
        return {
            "schema_version": API_VERSION,
            "id": identity,
            "ok": False,
            "error": {"code": code, "message": str(exc)},
        }
