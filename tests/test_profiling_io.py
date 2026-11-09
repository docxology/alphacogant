"""Real tracing/memory runs and bounded atomic I/O (no mocked filesystem)."""

import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

from alphacogant.api import Client
from alphacogant.io import parse_json, read_text, write_text
from alphacogant.profiling import profile_workload, source_fingerprint


def _assert_timing_unavailable(result):
    timing = result["timing"]
    assert timing["status"] == "unavailable"
    assert timing["unprofiled_seconds"] == []
    assert timing["median_seconds"] is None
    assert timing["min_seconds"] is None
    assert timing["max_seconds"] is None
    assert timing["reasons"]
    assert result["tracing"]["hotspots"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("workload", ["bootstrap", "inference", "efe"])
def test_seeded_profile_has_separate_timings_safe_hotspots_and_memory(workload, tmp_path):
    # -S avoids site/coverage startup hooks; -E ignores inherited Python tracing
    # settings. Explicit paths retain this checkout and its installed NumPy.
    code = """
import json
import sys
sys.path[:] = json.loads(sys.argv[1])
from alphacogant.profiling import profile_workload
result = profile_workload(workload=sys.argv[2], samples=2, n=2, horizon=2)
print(json.dumps(result, allow_nan=False))
"""
    child = subprocess.run(
        [
            sys.executable, "-E", "-S", "-B", "-c", code,
            json.dumps([str(Path(p or ".").resolve()) for p in sys.path]), workload,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    timing = result["timing"]
    assert timing["status"] == "ok"
    assert timing["reasons"] == []
    assert timing["instrumentation"] == {
        "sys_settrace": False, "tracemalloc": False, "sys_monitoring_tools": []
    }
    samples = timing["unprofiled_seconds"]
    assert len(samples) == 2
    assert all(t > 0 for t in samples)
    assert timing["median_seconds"] == sum(samples) / 2
    assert timing["min_seconds"] == min(samples)
    assert timing["max_seconds"] == max(samples)
    assert result["source_sha256"] == source_fingerprint()
    assert result["memory"]["status"] == "measured"
    assert result["memory"]["peak_bytes"] > 0
    assert result["tracing"]["hotspots"]
    assert all(not r["file"].startswith("/") for r in result["tracing"]["hotspots"])
    expected_module = "profiling.tracing" if sys.version_info >= (3, 15) else "cProfile"
    assert result["tracing"]["module"] == expected_module
    repeat = Client().call(
        "profile.run", {"workload": workload, "samples": 1, "n": 2, "horizon": 2, "memory": False}
    )
    assert repeat["result_sha256"] == result["result_sha256"]
    assert repeat["memory"]["status"] == "disabled"
    json.dumps(repeat, allow_nan=False)


@pytest.mark.parametrize(
    "params",
    [
        {"workload": "unknown"},
        {"samples": True},
        {"samples": 16},
        {"n": 2049},
        {"n": 1024, "horizon": 256},
        {"horizon": 257},
        {"seed": True},
        {"seed": -1},
        {"memory": 1},
    ],
)
def test_profile_admission(params):
    with pytest.raises(ValueError):
        profile_workload(**params)


def test_profile_preserves_existing_profiler():
    previous = sys.getprofile()

    def hook(*args):
        return None

    sys.setprofile(hook)
    try:
        with pytest.raises(ValueError, match="already active"):
            profile_workload(n=1, samples=1, horizon=1)
        assert sys.getprofile() is hook
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("memory", [False, True])
def test_profile_preserves_existing_tracemalloc(memory):
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start(7)
    marker = bytearray(8192)
    traceback = tracemalloc.get_object_traceback(marker)
    traceback_limit = tracemalloc.get_traceback_limit()
    _, previous_peak = tracemalloc.get_traced_memory()
    try:
        result = profile_workload(n=1, samples=1, horizon=1, memory=memory)
        _assert_timing_unavailable(result)
        assert result["timing"]["instrumentation"]["tracemalloc"] is True
        assert result["memory"]["status"] == ("unavailable" if memory else "disabled")
        assert tracemalloc.is_tracing()
        assert tracemalloc.get_traceback_limit() == traceback_limit
        assert traceback is not None
        assert tracemalloc.get_object_traceback(marker) == traceback
        assert tracemalloc.get_traced_memory()[1] >= previous_peak
    finally:
        if not was_tracing:
            tracemalloc.stop()


@pytest.mark.parametrize("memory", [False, True])
def test_profile_preserves_existing_trace_hook(memory):
    previous = sys.gettrace()
    workload_calls = 0

    def hook(frame, event, arg):
        nonlocal workload_calls
        if (
            event == "call"
            and frame.f_code.co_name == "run"
            and frame.f_code.co_filename == profile_workload.__code__.co_filename
        ):
            workload_calls += 1
        return hook

    sys.settrace(hook)
    try:
        result = profile_workload(n=1, samples=3, horizon=1, memory=memory)
        _assert_timing_unavailable(result)
        assert result["timing"]["instrumentation"]["sys_settrace"] is True
        assert sys.gettrace() is hook
        # Only warmup, profiler and (if requested) memory passes, no wall samples.
        assert workload_calls == 2 + int(memory and result["memory"]["status"] == "measured")
    finally:
        sys.settrace(previous)


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires sys.monitoring")
@pytest.mark.parametrize("active_events", [False, True])
def test_profile_preserves_registered_monitoring_tool(active_events):
    monitoring = sys.monitoring
    tool_id = next((i for i in reversed(range(6)) if monitoring.get_tool(i) is None), None)
    if tool_id is None:
        pytest.skip("all monitoring tool IDs belong to the caller")
    monitoring.use_tool_id(tool_id, "alphacogant-test")
    calls = 0

    def on_start(code, offset):
        nonlocal calls
        if (
            code.co_name == "run"
            and code.co_filename == profile_workload.__code__.co_filename
        ):
            calls += 1

    event = monitoring.events.PY_START
    monitoring.register_callback(tool_id, event, on_start)
    events = event if active_events else 0
    monitoring.set_events(tool_id, events)
    try:
        result = profile_workload(n=1, samples=3, horizon=1, memory=False)
        _assert_timing_unavailable(result)
        assert {"id": tool_id, "name": "alphacogant-test"} in (
            result["timing"]["instrumentation"]["sys_monitoring_tools"]
        )
        assert monitoring.get_tool(tool_id) == "alphacogant-test"
        assert monitoring.get_events(tool_id) == events
        assert calls == (2 if active_events else 0)
    finally:
        monitoring.set_events(tool_id, 0)
        monitoring.register_callback(tool_id, event, None)
        monitoring.free_tool_id(tool_id)


@pytest.mark.parametrize(
    "text", ['{"a":1,"a":2}', "NaN", "Infinity", "1e999", "[" * 2000 + "]" * 2000]
)
def test_json_rejects_duplicate_nonfinite_and_over_nested_content(text):
    with pytest.raises(ValueError):
        parse_json(text)


def test_atomic_outputs_do_not_clobber_and_preserve_modes(tmp_path):
    path = tmp_path / "nested" / "result.json"
    write_text('{"a":1}', path)
    assert parse_json(read_text(path)) == {"a": 1}
    assert path.read_bytes().endswith(b"\n")
    path.chmod(0o640)
    with pytest.raises(FileExistsError):
        write_text("new", path)
    assert parse_json(read_text(path)) == {"a": 1}
    write_text('{"b":2}', path, overwrite=True)
    assert path.stat().st_mode & 0o777 == 0o640
    assert parse_json(read_text(path)) == {"b": 2}
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        write_text("new", link, overwrite=True)
    assert not list(path.parent.glob(".alphacogant-*"))
    with pytest.raises(ValueError, match="exceeds"):
        read_text(path, limit=2)
    binary = tmp_path / "bad.txt"
    binary.write_bytes(b"\xff")
    with pytest.raises(UnicodeError):
        read_text(binary)
    assert parse_json('{"a":[1,2.5,true,null]}') == {"a": [1, 2.5, True, None]}
