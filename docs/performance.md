# Collect instrumentation-aware, source-bound performance evidence

The development interpreter is the one `.python-version` pins. The package still
supports Python 3.11+; CI exercises 3.11, 3.12, 3.13, 3.14 and 3.15. Select an alternate
interpreter with `uv sync --python 3.11 --locked --extra dev`.

## Collect matched workload timings with source and runtime provenance

```bash
uv sync --locked --extra dev
uv run --locked alphacogant profile --workload bootstrap --samples 5 \
  --n 32 --horizon 12 --seed 7 -o output/bootstrap-profile.json
uv run --locked alphacogant profile --workload inference --horizon 32 \
  --samples 5 -o output/inference-profile.json
```

In a fresh, uninstrumented process the workload runs once to warm up, records
raw wall-time samples, then executes a separate deterministic tracing run and an
optional separate `tracemalloc` run. `--no-memory` disables the last pass.
`bootstrap` uses `n` paired samples of `horizon` cycles; `inference` uses `horizon`
steps; `efe` scores all actions `n` times. Inputs have finite limits in `capabilities`.

Python 3.15 uses the standard library's
[`profiling.tracing`](https://docs.python.org/3.15/library/profiling.tracing.html).
Python 3.11–3.14 use `cProfile`. Hotspots are exported as bounded JSON records;
no pickle stats are loaded or exposed. Reports bind package-source SHA256,
version, runtime, NumPy, workload, seed and thread settings. Changed package
sources during a run invalidate it. Existing tracing profilers are preserved by
refusing nested profiling. Detected caller `sys.settrace`, `tracemalloc`, or
registered `sys.monitoring` tools disable wall sampling: `timing.status` is
`unavailable`, samples are empty, median/min/max are `null`, and reasons plus
instrumentation metadata are retained. Clean timing reports `status: ok`.
Caller instrumentation is never suspended/reset; an existing tracemalloc
session also makes the report's memory measurement unavailable.

Detection describes the collection entry state, not every possible external
instrument. Source fingerprints cover Python files on disk, not already-imported
code; collect comparable evidence from fresh processes/stable installations.
Recorded thread environment variables do not prove the loaded BLAS build or its
actual thread count. Tracemalloc excludes some native NumPy allocations.
Shared scheduling, thermal state and BLAS choices affect timings: compare matched
workloads/environments and retain raw samples, never wall-time CI thresholds.
CI retains the Python 3.15 profile separately from numerical acceptance.

## Preserve model semantics while permitting floating-point roundoff

`efe_vector` validates a public belief once, predicts all six actions in a batch,
and broadcasts masked KL across the action axis. KL uses log probabilities to
avoid ratio overflow for subnormal prior mass. Public single-action evaluation
remains available as a differential reference. Internal bootstrap trajectories
skip repeated admission checks but preserve normalization operations. Dirichlet
parameters are hoisted outside the sample loop; sample/channel draw order and
explicit generators remain unchanged. No mutable model-result cache is used.

Backend imports defer plotting until `alphacogant.plot_style` is accessed or a
visualization module is imported. Existing top-level plotting access is retained.
This also removes plotting backend initialization from agent and CLI startup.

Differential tests cover randomized models/priors, sparse likelihoods and point
mass priors, paired bootstrap rates and RNG continuation.
Rate-timeline corrections in 1.2.2 intentionally change manuscript diagnostics;
regenerate rather than require equality with the old, truncated-horizon rates.
Batched contractions can change the final bits of floating-point values; exact
trace replay requires a matching engine/numerical environment.

## Treat external sampling availability separately from measured cost

Python 3.15 also provides
[`profiling.sampling`](https://docs.python.org/3.15/library/profiling.sampling.html).
For a bounded external sampling session on a supported host:

```bash
uv run --locked python -m profiling.sampling run -d 3 --jsonl \
  -o output/sampling.jsonl -m alphacogant bootstrap --n 256 --horizon 32
```

The historical 1.2.1 macOS sampling attempt was unavailable because process
memory access was denied. No elevated access was requested. Deterministic tracing,
unprofiled timings and memory reports were collected successfully; sampling
absence is recorded separately and is not a zero-cost observation.

## Historical 1.2.1 release measurement

A matched local Python 3.15.0 / NumPy 2.5.4 macOS arm64 run, with all three
recorded BLAS/OpenMP thread settings set to 1 and five wall samples per workload,
compared v1.1.1 against v1.2.1:

| Workload | v1.1.1 median | v1.2.1 median | Ratio |
| --- | ---: | ---: | ---: |
| 32-sample, 12-cycle bootstrap | 76.75 ms | 33.69 ms | 2.28× |
| 32-step inference | 5.76 ms | 4.75 ms | 1.21× |
| Fresh-process package import | 192.93 ms | 59.67 ms | 3.23× |

These measurements describe this host and workload. They are not CI thresholds or
performance guarantees. The profiling command above produces source-bound raw
samples for new environments.

CI requires wheels for NumPy, Matplotlib, Pillow, ContourPy and KiwiSolver.
Version 1.2.1 refreshes these locks for Linux CPython 3.15 rather than compiling
native plotting dependencies with missing system image headers.
