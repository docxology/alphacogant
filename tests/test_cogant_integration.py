"""Optional execution against the real COGANT package; no substitute runtime."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from alphacogant.bridge.gnn import joint_matrices, model_to_cogant_gnn


def test_current_cogant_parses_all_coefficients_and_executes(model):
    pytest.importorskip("cogant", reason="install COGANT to run the upstream integration check")
    from cogant.reverse.parser import parse_gnn
    from cogant.runtime import InferenceOptions, run_gnn_inference

    source = model_to_cogant_gnn(model)
    parsed = parse_gnn(source)
    assert not parsed.degraded, parsed.diagnostics
    assert parsed.matrix_source == "fenced"
    matrices = joint_matrices(model)
    for name, expected in matrices.items():
        np.testing.assert_array_equal(getattr(parsed, name), expected)
    trace = run_gnn_inference(source, InferenceOptions(steps=3, observations=(0, 4, 8)))
    assert trace.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert len(trace.trace) == 3
    for name, expected in matrices.items():
        np.testing.assert_array_equal(getattr(trace.matrices, name), expected)
    for record in trace.trace:
        assert len(record.posterior) == 32
        np.testing.assert_allclose(sum(record.posterior), 1.0, atol=1e-14)
        np.testing.assert_allclose(sum(record.predicted_belief), 1.0, atol=1e-14)
        assert len(record.policy_scores) == 6
