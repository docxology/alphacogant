"""Real tracing/memory runs and bounded atomic I/O (no mocked filesystem)."""

import json
import sys
import tracemalloc

import pytest

from alphacogant.api import Client
from alphacogant.io import parse_json, read_text, write_text
from alphacogant.profiling import profile_workload, source_fingerprint


@pytest.mark.parametrize("workload", ["bootstrap", "inference", "efe"])
def test_seeded_profile_has_separate_timings_safe_hotspots_and_memory(workload):
    result = profile_workload(workload=workload, samples=2, n=2, horizon=2)
    assert len(result["timing"]["unprofiled_seconds"]) == 2
    assert all(t > 0 for t in result["timing"]["unprofiled_seconds"])
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
    json.dumps(result, allow_nan=False)


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


def test_profile_preserves_existing_instrumentation():
    sys.setprofile(lambda *args: None)
    try:
        with pytest.raises(ValueError, match="already active"):
            profile_workload(n=1, samples=1, horizon=1)
        assert sys.getprofile() is not None
    finally:
        sys.setprofile(None)
    tracemalloc.start()
    try:
        result = profile_workload(n=1, samples=1, horizon=1)
        assert result["memory"]["status"] == "unavailable"
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


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
