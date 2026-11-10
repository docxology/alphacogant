# Call bounded Python, JSONL and CLI research methods

`alphacogant.api.Client` and `alphacogant request` share API version `1.0` and
one parameter-admission registry. Run `alphacogant capabilities` to discover
method descriptions, JSON parameter schemas, defaults and limits. This is a
finite local protocol: methods compute over supplied content and built-in model
arrays. They do not execute commands, load arbitrary files or plugins, write
artifacts, or call the network.

## Use the Python client or versioned request envelope

```python
from alphacogant.api import Client

client = Client()
health = client.call("diagnostics")
trajectory = client.call("simulation.run", {
    "policy": "stochastic", "seed": 7, "horizon": 12,
})
statistics = client.call("statistics.bootstrap", {
    "seed": 7, "n": 32, "horizon": 12, "concentration": 12.0,
})
```

`Client.call` returns result data and raises ordinary admission errors. Omitted
parameters or Python `params=None` mean an empty parameter object. Use
`handle_request` for the same response envelope as the JSONL transport; there,
omitted `params` means `{}`, but explicitly supplied `params` must be an object
(JSON `null` is rejected).

## Stream bounded JSONL frames and discover method schemas

```bash
printf '%s\n' \
  '{"id":"health","method":"diagnostics"}' \
  '{"id":"run","method":"simulation.run","params":{"seed":7,"horizon":3}}' \
  | uv run --locked alphacogant request
```

Each line produces one flushed JSON response:

```json
{"schema_version":"1.0","id":"run","ok":true,"result":{"summary":{},"trajectory":{}}}
```

The example abbreviates result objects. Errors use `ok: false` and an `error`
object with `code` and `message`. An unknown method uses `method_not_found`;
admission, input and computation failures use `invalid_request`. Failed frames
do not prevent subsequent valid frames from running. Exit status is 2 if any
frame failed, otherwise 0. An oversized frame closes the stream after its error.
At most 1000 frames are accepted. `alphacogant.api.MAX_REQUEST_BYTES` is
32,000,000 UTF-8 bytes per complete request line, including JSON escaping and the
newline, matching the file-backed JSON budget. Source text retains its separate
1,000,000-byte limit. The frame budget accommodates a maximum-source, 1000-step
trace plus its source in a subsequent `inference.verify` request.
JSON nesting is bounded at 64 levels; duplicate keys, NaN/Infinity literals and
overflowing floating-point literals are rejected. Stdout contains responses only.
Malformed GNN assignments and unrepresentable integer coefficients produce
ordinary errors, so the next valid frame can still run.

| Method | Result |
| --- | --- |
| `capabilities` | Current version, method schemas and limits. |
| `diagnostics` | Runtime/distribution identity, source fingerprint and real model round-trip. |
| `model.validate` | Finite GNN admission and parameter shapes; `source` is required text. |
| `model.export` | Full GNN `source`, optionally with joint COGANT matrices (`cogant: true`). |
| `inference.run` | Source-bound filter/score/predict trace; source text required. |
| `inference.verify` | Strict source/trace replay, yielding `verified: true`. |
| `simulation.run` | Open-loop allocation trajectory and summary. |
| `statistics.bootstrap` | Paired t-RSI/create/decay statistics. |
| `profile.run` | Bounded seeded timing, tracing and memory report. |

Unknown parameters and booleans supplied as integer bounds are rejected. Seeds
are explicit defaults (7) and each computation creates its own NumPy generator.
Agent inference/simulation is limited to 1000 steps. Bootstrap/profile limits
include `n <= 2048`, `horizon <= 256`, and `n * horizon <= 65536`; profiling admits
at most 15 timing samples. Choose smaller requests for interactive response.
The full manuscript uses its separate published sample count through the Python
research APIs, which retain their existing contracts.

Mathematically infinite t-RSI results are encoded as the JSON strings `Infinity`
or `-Infinity`; NaN, if present in a numerical result, is encoded as `NaN`.
Ordinary finite results remain JSON numbers. Shared scalar/batch action scoring
rejects nonfinite computed preference values before any policy can select an
action or publish a scored trajectory. This applies to inference, simulation
and bootstrap consumers, not just inference traces. Intentional t-RSI infinities
are unaffected. Capability discovery describes the
general numerical encoding.
Joint-matrix export also rejects unrepresentable combined preference scores.
Simulation also rejects unrepresentable cumulative pragmatic/epistemic totals
at the first offending cycle, even if each individual score is finite.
Profiling may return unavailable wall timing with empty samples and nullable
statistics under caller instrumentation; see [performance](performance.md).

## Use CLI envelopes and publish files without partial writes

`validate`, `export`, `infer`, and `verify-trace` retain their default output
formats. `simulate`, `bootstrap`, `doctor`, `capabilities`, and `profile` emit
JSON. `--json` wraps results and runtime errors in the same versioned success/error
envelope; argparse usage errors remain on stderr. `--version` reports the installed
package version.

Source reads are bounded to 1 MB; observation/trace JSON reads to 32 MB. Commands
that write accept `-o` and explicit `--overwrite`. They render a complete temporary
file before atomically linking or replacing the destination. Exclusive writes
cannot clobber another writer; overwrite preserves the existing mode. Symlink
outputs are rejected. This protects completed artifacts from partial writes;
it does not add crash-durable directory fsync guarantees.

Inference traces replay exact floating-point calculations. Keep the producing
engine and numerical runtime when replaying archived traces; a different engine
version or array kernel may fail exact verification. EFE batching can differ from
scalar evaluation by roundoff while preserving the numerical model. See
[performance](performance.md) and [bridge scope](cogant_bridge.md).

The source-owned manuscript renderer is a separate trusted local script, not an
agent method. It does not add file/process/network capabilities to API `1.0` or
introduce task, backlog or dashboard state protocols.
