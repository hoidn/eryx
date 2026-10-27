"""Regression checks for the documented simulation and plotting entry points."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

import run_torch
import visualize_diffuse
import eryx.visualization as visualization


def test_npz_visualization_uses_intensity_and_recorded_shape(tmp_path, monkeypatch):
    shape = (2, 3, 4)
    intensity = np.arange(24.0)
    path = tmp_path / "results.npz"
    np.savez(path, intensity=intensity, q_vectors=np.zeros((24, 3)), map_shape=shape)
    render = Mock(return_value=("plot", "volume"))
    monkeypatch.setattr(visualization, "_has_k3d", True)
    monkeypatch.setattr(visualization, "create_clipped_visualization", render, raising=False)

    assert visualization.visualize_diffuse_3d(str(path)) == ("plot", "volume")
    np.testing.assert_array_equal(render.call_args.args[0], intensity.reshape(shape))


def test_flat_volume_without_shape_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "flat.npy"
    np.save(path, np.arange(27.0))
    monkeypatch.setattr(visualization, "_has_k3d", True)
    monkeypatch.setattr(visualization, "create_clipped_visualization", Mock(), raising=False)
    with pytest.raises(ValueError, match="map_shape"):
        visualization.visualize_diffuse_3d(str(path))


def test_reference_shape_accepts_numpy_integers(tmp_path):
    path = tmp_path / "reference.npz"
    np.savez(path, map_shape=(2, 3, 4))
    assert visualize_diffuse.get_map_shape_from_reference([path]) == (2, 3, 4)


def test_runner_propagates_initialization_failure():
    with patch("eryx.models_torch.OnePhonon", side_effect=ValueError("invalid PDB")):
        with pytest.raises(ValueError, match="invalid PDB"):
            run_torch.run_torch(device="cpu")


def test_cuda_retry_explicitly_selects_cpu():
    with patch("eryx.models_torch.OnePhonon", side_effect=[
        RuntimeError("CUDA out of memory"), ValueError("stop after CPU selection")
    ]) as constructor:
        with pytest.raises(ValueError, match="CPU selection"):
            run_torch.run_torch(device="cuda")
    assert [call.kwargs["device"].type for call in constructor.call_args_list] == ["cuda", "cpu"]
