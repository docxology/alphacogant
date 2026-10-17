"""Standalone CLI for finite GNN validation, export, and inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alphacogant.bridge.gnn import model_from_gnn, model_to_cogant_gnn, model_to_gnn
from alphacogant.bridge.inference import run_inference, verify_inference
from alphacogant.model.generative_model import default_model


def _write(text: str, output: Path | None, overwrite: bool) -> None:
    if output is None:
        print(text, end="" if text.endswith("\n") else "\n")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    for command in (export, infer):
        command.add_argument("-o", "--output", type=Path)
        command.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            model_from_gnn(args.source.read_bytes().decode("utf-8"))
            print("Valid AlphaCOGANT finite GNN profile.")
        elif args.command == "export":
            model = (
                default_model()
                if args.source is None
                else model_from_gnn(args.source.read_bytes().decode("utf-8"))
            )
            render = model_to_cogant_gnn if args.cogant else model_to_gnn
            _write(render(model), args.output, args.overwrite)
        else:
            source = args.source.read_bytes().decode("utf-8")
            if args.command == "verify-trace":
                trace = json.loads(args.trace.read_text(encoding="utf-8"))
                verify_inference(source, trace)
                print("Inference trace verified by replay.")
            else:
                observations = (
                    None
                    if args.observations is None
                    else json.loads(args.observations.read_text(encoding="utf-8"))
                )
                steps = (
                    args.steps
                    if args.steps is not None
                    else (8 if observations is None else len(observations))
                )
                trace = run_inference(source, steps=steps, observations=observations)
                _write(
                    json.dumps(trace, indent=2, allow_nan=False) + "\n", args.output, args.overwrite
                )
    except (OSError, ValueError, TypeError) as exc:
        print(f"alphacogant: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
