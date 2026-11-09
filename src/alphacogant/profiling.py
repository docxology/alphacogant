"""Bounded, seeded performance evidence with separate timing, tracing and memory runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import pstats
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from alphacogant.bridge.gnn import model_to_gnn
from alphacogant.bridge.inference import run_inference
from alphacogant.efe.free_energy import efe_vector
from alphacogant.model.generative_model import belief_prior, default_model
from alphacogant.trsi.t_rsi import bootstrap_t_rsi
from alphacogant.validation import positive_integer


def source_fingerprint(*, root: Path | None = None) -> str:
    """Fingerprint disk sources, not already-imported code; optionally compare a checkout."""
    root = Path(__file__).resolve().parent if root is None else Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def profile_workload(
    *,
    workload: str = "bootstrap",
    samples: int = 5,
    n: int = 32,
    horizon: int = 12,
    seed: int = 7,
    memory: bool = True,
) -> dict[str, object]:
    """Profile a fixed default-model workload without accepting executable code.

    Python 3.15 uses ``profiling.tracing``; older supported runtimes use its
    ``cProfile`` predecessor. Wall timings are separate from our tracing and
    tracemalloc passes, and unavailable under detected caller instrumentation.
    Reports contain JSON data, never executable pickle stats.
    """
    if workload not in {"bootstrap", "inference", "efe"}:
        raise ValueError("workload must be bootstrap, inference or efe.")
    for name, value, limit in (("samples", samples, 15), ("n", n, 2048), ("horizon", horizon, 256)):
        positive_integer(value, name)
        if value > limit:
            raise ValueError(f"{name} must be <= {limit}.")
    if n * horizon > 65_536:
        raise ValueError("n * horizon must be <= 65536.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1].")
    if not isinstance(memory, bool):
        raise ValueError("memory must be boolean.")
    if sys.getprofile() is not None:
        raise ValueError("A tracing profiler is already active; run profiling separately.")
    monitoring_tools = []
    if hasattr(sys, "monitoring"):
        for tool_id in range(6):  # sys.monitoring reserves tool IDs 0 through 5
            name = sys.monitoring.get_tool(tool_id)
            if name is not None:
                monitoring_tools.append({"id": tool_id, "name": name})
    instrumentation = {
        "sys_settrace": sys.gettrace() is not None,
        "tracemalloc": tracemalloc.is_tracing(),
        "sys_monitoring_tools": monitoring_tools,
    }
    timing_reasons = []
    if instrumentation["sys_settrace"]:
        timing_reasons.append("sys.settrace tracing is already active")
    if instrumentation["tracemalloc"]:
        timing_reasons.append("tracemalloc is already active")
    if monitoring_tools:
        timing_reasons.append("sys.monitoring tools are already registered")

    if sys.version_info >= (3, 15):
        from profiling import tracing as profiler_module
    else:
        import cProfile as profiler_module

    source_hash = source_fingerprint()
    model = default_model()
    belief = belief_prior(model)
    source = model_to_gnn(model)

    def run():
        if workload == "bootstrap":
            return bootstrap_t_rsi(model, belief, np.random.default_rng(seed), n, horizon)
        if workload == "inference":
            return run_inference(source, steps=horizon)
        return [[r.total for r in efe_vector(model, belief)] for _ in range(n)]

    reference = run()  # warm allocations and imports before measuring
    timings = []
    if not timing_reasons:
        for _ in range(samples):
            started = time.perf_counter()
            run()
            timings.append(time.perf_counter() - started)
    profiler = profiler_module.Profile()
    profiler.runcall(run)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in sorted(
        pstats.Stats(profiler).stats.items(), key=lambda item: item[1][3], reverse=True
    )[:30]:
        # Do not disclose installation/home paths in portable agent reports.
        name = (
            "alphacogant/" + filename.split("/alphacogant/")[-1]
            if ("/alphacogant/" in filename)
            else Path(filename).name
        )
        rows.append(
            {
                "file": name,
                "line": line,
                "function": function,
                "primitive_calls": primitive,
                "calls": calls,
                "self_seconds": own,
                "cumulative_seconds": cumulative,
            }
        )
    memory_result: dict[str, object] = {"status": "disabled"}
    if memory and tracemalloc.is_tracing():
        memory_result = {"status": "unavailable", "reason": "tracemalloc already active"}
    elif memory:
        tracemalloc.start()
        try:
            run()
            current, peak = tracemalloc.get_traced_memory()
            memory_result = {"status": "measured", "current_bytes": current, "peak_bytes": peak}
        finally:
            tracemalloc.stop()
    if source_fingerprint() != source_hash:
        raise ValueError("Package sources changed during profiling; repeat on stable sources.")
    from alphacogant import __version__

    return {
        "schema_version": "1.0",
        "package_version": __version__,
        "source_sha256": source_hash,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
            "threads": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "workload": {
            "name": workload,
            "samples": samples,
            "n": n,
            "horizon": horizon,
            "seed": seed,
        },
        "timing": {
            "status": "unavailable" if timing_reasons else "ok",
            "reasons": timing_reasons,
            "instrumentation": instrumentation,
            "unprofiled_seconds": timings,
            "median_seconds": statistics.median(timings) if timings else None,
            "min_seconds": min(timings) if timings else None,
            "max_seconds": max(timings) if timings else None,
        },
        "tracing": {"module": profiler_module.__name__, "hotspots": rows},
        "memory": memory_result,
        "result_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        "limitations": [
            "Timings are machine-specific; compare matched workloads and environments.",
            "This report's tracing and memory passes are separate from wall sampling; "
            "detected caller instrumentation disables wall sampling.",
            "External instrumentation not exposed by Python may be undetectable.",
            "The source fingerprint covers files on disk, not already-imported module code.",
            "tracemalloc excludes some native NumPy allocations.",
        ],
    }
