"""Standalone model, inference, diagnostics and agent CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alphacogant import __version__
from alphacogant.api import API_VERSION, MAX_REQUEST_BYTES, Client, handle_request, json_value
from alphacogant.bridge.inference import run_inference, verify_inference
from alphacogant.io import MAX_JSON_BYTES, parse_json, read_text, write_text


def _write(text: str, output: Path | None, overwrite: bool) -> None:
    if output is None:
        print(text, end="" if text.endswith("\n") else "\n")
    else:
        write_text(text, output, overwrite=overwrite)


def _json(data) -> str:
    return json.dumps(json_value(data), indent=2, allow_nan=False) + "\n"


def _stream() -> int:
    """One JSON request and response per line, with bounded frame and stream lengths."""
    failed = False
    for _ in range(1000):
        raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            return 2 if failed else 0
        try:
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError(f"Request exceeds {MAX_REQUEST_BYTES} bytes; stream closed.")
            response = handle_request(parse_json(raw.decode("utf-8")))
        except (ValueError, UnicodeError) as exc:
            response = {
                "schema_version": API_VERSION,
                "id": None,
                "ok": False,
                "error": {"code": "invalid_request", "message": str(exc)},
            }
        print(json.dumps(response, allow_nan=False), flush=True)
        failed |= not response["ok"]
        if len(raw) > MAX_REQUEST_BYTES:
            return 2  # never treat a remainder of the same oversized frame as a new request
    if sys.stdin.buffer.read(1):
        print(
            json.dumps(
                {
                    "schema_version": API_VERSION,
                    "id": None,
                    "ok": False,
                    "error": {
                        "code": "limit_exceeded",
                        "message": "At most 1000 requests per stream.",
                    },
                }
            ),
            flush=True,
        )
        return 2
    return 2 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"alphacogant {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate this project's finite GNN profile")
    validate.add_argument("source", type=Path)
    export = sub.add_parser("export", help="export complete live model coefficients")
    export.add_argument("--source", type=Path, help="alternate finite GNN model")
    export.add_argument("--cogant", action="store_true", help="include complete joint matrices")
    infer = sub.add_parser("infer", help="run a source-bound observation-conditioned trace")
    infer.add_argument("source", type=Path)
    infer.add_argument("--steps", type=int)
    infer.add_argument("--observations", type=Path, help="JSON array of [reward,loss] pairs")
    verify = sub.add_parser("verify-trace", help="replay and verify a saved trace")
    verify.add_argument("source", type=Path)
    verify.add_argument("trace", type=Path)
    simulate = sub.add_parser("simulate", help="record a seeded allocation trajectory")
    simulate.add_argument("--source", type=Path)
    simulate.add_argument("--horizon", type=int, default=12)
    simulate.add_argument(
        "--policy", choices=["greedy", "hold", "fund_theta", "stochastic"], default="greedy"
    )
    simulate.add_argument("--seed", type=int, default=7)
    bootstrap = sub.add_parser("bootstrap", help="paired create/decay bootstrap statistics")
    bootstrap.add_argument("--source", type=Path)
    bootstrap.add_argument("--n", type=int, default=32)
    bootstrap.add_argument("--horizon", type=int, default=12)
    bootstrap.add_argument("--seed", type=int, default=7)
    bootstrap.add_argument("--concentration", type=float, default=12.0)
    profile = sub.add_parser("profile", help="bounded performance evidence (Python 3.15 tracing)")
    profile.add_argument(
        "--workload", choices=["bootstrap", "inference", "efe"], default="bootstrap"
    )
    profile.add_argument("--samples", type=int, default=5)
    profile.add_argument("--n", type=int, default=32)
    profile.add_argument("--horizon", type=int, default=12)
    profile.add_argument("--seed", type=int, default=7)
    profile.add_argument("--no-memory", action="store_true")
    capabilities = sub.add_parser("capabilities", help="discover methods, schemas and limits")
    doctor = sub.add_parser("doctor", help="check runtime identity and model round-trip")
    sub.add_parser("request", help="process bounded JSONL requests from stdin")
    for command in (
        validate,
        export,
        infer,
        verify,
        simulate,
        bootstrap,
        profile,
        capabilities,
        doctor,
    ):
        command.add_argument("--json", action="store_true", help="JSON success/error envelope")
    for command in (export, infer, simulate, bootstrap, profile, capabilities, doctor):
        command.add_argument("-o", "--output", type=Path)
        command.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "request":
            return _stream()
        client = Client()
        source = read_text(args.source) if getattr(args, "source", None) is not None else None
        if args.command == "validate":
            result = client.call("model.validate", {"source": source})
            text = "Valid AlphaCOGANT finite GNN profile.\n"
        elif args.command == "export":
            params = {"cogant": args.cogant}
            if source is not None:
                params["source"] = source
            result = client.call("model.export", params)
            text = result["source"]
        elif args.command == "verify-trace":
            trace = parse_json(read_text(args.trace, limit=MAX_JSON_BYTES))
            verify_inference(source, trace)
            result = {"verified": True}
            text = "Inference trace verified by replay.\n"
        elif args.command == "infer":
            observations = (
                None
                if args.observations is None
                else parse_json(read_text(args.observations, limit=MAX_JSON_BYTES))
            )
            steps = (
                args.steps
                if args.steps is not None
                else (8 if observations is None else len(observations))
            )
            result = run_inference(source, steps=steps, observations=observations)
            text = _json(result)
        else:
            params = {
                key: value
                for key, value in vars(args).items()
                if key not in {"command", "json", "output", "overwrite", "source", "no_memory"}
            }
            if source is not None:
                params["source"] = source
            if args.command == "profile":
                params["memory"] = not args.no_memory
            method = {
                "simulate": "simulation.run",
                "bootstrap": "statistics.bootstrap",
                "profile": "profile.run",
                "capabilities": "capabilities",
                "doctor": "diagnostics",
            }[args.command]
            result = client.call(method, params)
            text = _json(result)
        if args.json:
            text = _json({"schema_version": API_VERSION, "id": None, "ok": True, "result": result})
        _write(text, getattr(args, "output", None), getattr(args, "overwrite", False))
        if args.command == "doctor" and not result["healthy"]:
            return 2
    except BrokenPipeError:
        return 0
    except (OSError, ValueError, TypeError, LookupError, RecursionError) as exc:
        if getattr(args, "json", False):
            print(
                _json(
                    {
                        "schema_version": API_VERSION,
                        "id": None,
                        "ok": False,
                        "error": {"code": "invalid_request", "message": str(exc)},
                    }
                ),
                end="",
            )
        else:
            print(f"alphacogant: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
