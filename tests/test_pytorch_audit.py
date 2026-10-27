"""Numerical regressions that do not depend on captured state logs."""

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np
import torch

from eryx.models_torch import OnePhonon
from eryx.pdb_torch import GaussianNetworkModel
from eryx.torch_utils import BrillouinZoneUtils, EigenOps


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def phonon_model(matrix):
    model = OnePhonon.__new__(OnePhonon)
    model.device = matrix.device
    model.real_dtype = torch.float64
    model.complex_dtype = torch.complex128
    model.n_asu = model.n_cell = 1
    model.id_cell_ref = 0
    model.n_atoms_per_asu = model.n_dof_per_asu = matrix.shape[0]
    model.use_arbitrary_q = True
    model.kvec = matrix.real.new_tensor([[0, 0, 0], [0.1, 0, 0], [0, 0, 0]])
    model.Linv = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    model.crystal = SimpleNamespace()
    model.gnm = SimpleNamespace()
    model.compute_hessian = lambda: matrix.reshape(1, matrix.shape[0], 1, 1, matrix.shape[1])
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_zero_mode", [False, True])
@pytest.mark.parametrize("pdos_mode", [None, "direct", "thermal"])
def test_phonon_population_gradient_matches_finite_difference(device, with_zero_mode, pdos_mode):
    def populations(scale):
        diagonal = scale.new_tensor([0.0 if with_zero_mode else 1.0, 3.0]) * scale
        model = phonon_model(torch.diag(diagonal).to(torch.complex128))
        if pdos_mode:
            model.pdos_mode = pdos_mode
            model.temperature_k = 300.0
            model.pdos_omega = scale.new_tensor([0.0, 1.0, 4.0]) * 1e14
            model.pdos_density = scale.new_tensor([0.5, 1.5, 2.0])
        model.compute_gnm_phonons()
        if with_zero_mode:
            assert torch.isnan(model.Winv[:, 0]).all()
        return model.Winv.real[:, 1:].sum()

    scale = torch.tensor(2.0, dtype=torch.float64, device=device, requires_grad=True)
    result = populations(scale)
    derivative, = torch.autograd.grad(result, scale)
    step = 1e-5
    expected = (populations(scale.detach() + step) - populations(scale.detach() - step)) / (2 * step)
    torch.testing.assert_close(derivative, expected, atol=1e-8, rtol=1e-5)


def test_phonons_do_not_repeat_eigensolves():
    matrix = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.complex128))
    model = phonon_model(matrix)
    with patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
        model.compute_gnm_phonons()
    # A single batched solve or one solve per unique k-vector is sufficient.
    assert eigh.call_count <= 2
    torch.testing.assert_close(model.Winv.real, torch.tensor([[0.5, 0.2]] * 3, dtype=torch.float64))


def test_phonon_gradient_with_grid_kvec_and_degenerate_modes():
    scale = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    model = phonon_model(torch.eye(3, dtype=torch.complex128) * scale)
    model.n_cell = 2
    model.crystal = SimpleNamespace(id_to_hkl=lambda i: i,
                                    get_unitcell_origin=lambda _: torch.tensor([1.0, 0, 0], dtype=torch.float64))
    model.compute_hessian = lambda: torch.stack([
        (torch.eye(3, dtype=torch.complex128) * scale).reshape(1, 3, 1, 3),
        (torch.eye(3, dtype=torch.complex128) * scale * 0.2).reshape(1, 3, 1, 3),
    ], dim=2)
    model.kvec.requires_grad_()
    model.compute_gnm_phonons()
    loss = model.Winv.real.sum()
    expected = -loss.detach() / scale.detach()
    loss.backward()
    torch.testing.assert_close(scale.grad, expected)


@pytest.mark.parametrize("pdos_mode", ["direct", "thermal"])
def test_pdos_import_uses_exported_frequency_units(tmp_path, pdos_mode):
    model = phonon_model(torch.diag(torch.tensor([0.01, 0.04], dtype=torch.complex128)))
    model.compute_gnm_phonons()
    frequencies = model.get_frequencies_thz()
    expected = np.array([0.4, 1.2])
    path = tmp_path / "pdos.dat"
    np.savetxt(path, np.column_stack([np.unique(frequencies), expected]))
    model.pdos_path = str(path)
    model.pdos_mode = pdos_mode
    model.temperature_k = 300.0
    model._load_and_prepare_pdos()
    model.compute_gnm_phonons()
    torch.testing.assert_close(model.Winv.real, torch.tensor([expected.tolist()] * 3, dtype=torch.float64))
    np.testing.assert_allclose(model.get_frequencies_thz(), frequencies)
    histogram, edges = np.histogram(frequencies, bins=3, density=True)
    np.testing.assert_allclose(model.generate_pdos(bins=3), np.column_stack([(edges[:-1] + edges[1:]) / 2, histogram]))


def test_pdos_interpolation_clamps_outside_data():
    model = OnePhonon.__new__(OnePhonon)
    model.pdos_omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    model.pdos_density = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64)
    result = model._differentiable_interp(torch.tensor([-1.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(result, torch.tensor([2.0, 1.0], dtype=torch.float64))


def test_phonon_eigensolver_failure_is_not_fabricated_data():
    model = phonon_model(torch.eye(2, dtype=torch.complex128))
    with patch("torch.linalg.eigh", side_effect=torch.linalg.LinAlgError("did not converge")):
        with pytest.raises(torch.linalg.LinAlgError, match="did not converge"):
            model.compute_gnm_phonons()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3, 3)])
def test_batched_pinv_rectangular_and_gradients(device, shape):
    generator = torch.Generator().manual_seed(42)
    matrix = torch.randn(2, *shape, generator=generator, dtype=torch.complex128).to(device).requires_grad_()
    gnm = GaussianNetworkModel()
    gnm.device = torch.device(device)
    actual = gnm._batched_pinv(matrix)
    expected = torch.linalg.pinv(matrix, rtol=1e-10)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual.abs().square().sum(), matrix, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.abs().square().sum(), matrix)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_inverse_failure_is_not_identity():
    gnm = GaussianNetworkModel()
    gnm.device = torch.device("cpu")
    gnm.n_cell = gnm.n_asu = 1
    gnm.id_cell_ref = 0
    hessian = torch.eye(2, dtype=torch.complex128).reshape(1, 2, 1, 1, 2)
    with patch.object(gnm, "_batched_pinv", side_effect=RuntimeError("did not converge")), \
         patch("torch.linalg.pinv", side_effect=RuntimeError("did not converge")):
        with pytest.raises(RuntimeError, match="did not converge"):
            gnm.compute_Kinv(hessian, torch.zeros(1, 3))


def test_pinv_gradient_at_repeated_singular_values():
    matrix = torch.eye(3, dtype=torch.complex128).unsqueeze(0).requires_grad_()
    gnm = GaussianNetworkModel()
    gnm.device = matrix.device
    gnm._batched_pinv(matrix).real.sum().backward()
    torch.testing.assert_close(matrix.grad, -torch.ones_like(matrix))


def test_hessian_uses_updated_spring_parameters():
    model = phonon_model(torch.eye(6, dtype=torch.complex128))
    del model.compute_hessian
    model.n_atoms_per_asu = 2
    model.Amat = torch.eye(6, dtype=torch.float64).unsqueeze(0)
    model.gamma_intra = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    model.gamma_inter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    model.gamma_tensor = model.gamma_intra.reshape(1, 1, 1) * 1.0
    model.gnm.asu_neighbors = [[[[[0, 1], [0, 1]]]]]
    first = model.compute_hessian()
    first.real.square().sum().backward()
    with torch.no_grad():
        model.gamma_intra.mul_(2)
    second = model.compute_hessian()
    torch.testing.assert_close(second, 2 * first)
    second.real.square().sum().backward()
    assert torch.isfinite(model.gamma_intra.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("decomposition", ["svd", "eigh"])
def test_complex_decompositions_reconstruct_input(device, decomposition):
    matrix = torch.tensor([[4, 1 + 2j], [1 - 2j, 3]], dtype=torch.complex128, device=device)
    regularized = matrix + 1e-10 * torch.eye(2, dtype=matrix.dtype, device=device)
    if decomposition == "svd":
        u, s, v = EigenOps.svd_decomposition(matrix)
        torch.testing.assert_close((u * s) @ v.mH, regularized)
    else:
        w, v = EigenOps.eigen_decomposition(matrix)
        torch.testing.assert_close((v * w) @ v.mH, regularized)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("single", [False, True])
def test_brillouin_mapping_for_skew_cell(device, single):
    reciprocal = torch.tensor([[0.2, -0.08, 0.03], [0, 0.15, -0.02], [0, 0, 0.1]], dtype=torch.float64, device=device)
    hkl = reciprocal.new_tensor([[0.2, -0.1, 0.3], [1.2, -2.1, 3.3]])
    if single:
        hkl = hkl[0]
    q = 2 * torch.pi * (hkl @ reciprocal)
    expected = reciprocal.new_tensor([0.2, -0.1, 0.3]) @ reciprocal
    actual = BrillouinZoneUtils.map_q_to_k_bz(q, reciprocal)
    torch.testing.assert_close(actual, expected.expand_as(actual))


@pytest.mark.parametrize("device", DEVICES)
def test_brillouin_boundary_roundoff_uses_same_representative(device):
    reciprocal = torch.diag(torch.tensor([0.2, 0.15, 0.1], dtype=torch.float64, device=device))
    hkl = reciprocal.new_tensor([[0.5 - 1e-15, 0, 0], [-0.5 + 1e-15, 0, 0]])
    actual = BrillouinZoneUtils.map_q_to_k_bz(2 * torch.pi * (hkl @ reciprocal), reciprocal)
    expected = reciprocal.new_tensor([[-0.1, 0, 0], [-0.1, 0, 0]])
    torch.testing.assert_close(actual, expected)


def intensity_model(device, n_points=5):
    model = OnePhonon.__new__(OnePhonon)
    model.device = torch.device(device)
    model.real_dtype = torch.float64
    model.complex_dtype = torch.complex128
    model.use_arbitrary_q = True
    model.n_asu = model.n_atoms_per_asu = model.n_processes = 1
    model.n_dof_per_asu = 3
    model.q_grid = (torch.arange(n_points * 3, dtype=torch.float64, device=device).reshape(n_points, 3) / 10 + 0.1).requires_grad_()
    model.res_mask = torch.arange(n_points, device=device) % 3 != 1
    model.ADP = torch.tensor([0.2], dtype=torch.float64, device=device, requires_grad=True)
    xyz = torch.tensor([[0.3, 0.7, -0.2]], dtype=torch.float64, device=device)
    model.crystal = SimpleNamespace(get_asu_xyz=lambda _: xyz)
    model.model = SimpleNamespace(ff_a=[torch.zeros(1, 4, device=device)],
                                  ff_b=[torch.ones(1, 4, device=device)],
                                  ff_c=[torch.ones(1, device=device)])
    model.Amat = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0)
    model.V = torch.tensor([[1, 0.2j, 0.3], [0.1, 1, 0.4j], [0.5j, 0.2, 1]], dtype=torch.complex128, device=device).expand(n_points, -1, -1).clone().requires_grad_()
    model.Winv = torch.tensor([0.5, 0.2, 0.1], dtype=torch.complex128, device=device).expand(n_points, -1).clone().requires_grad_()
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [-1, 0, 2])
@pytest.mark.parametrize("n_points", [0, 1, 5])
def test_arbitrary_intensity_matches_scalar_formula(device, rank, n_points):
    model = intensity_model(device, n_points)
    actual = model.apply_disorder(rank=rank)
    assert torch.isnan(actual[~model.res_mask]).all()
    expected = []
    for i in torch.where(model.res_mask)[0]:
        q = model.q_grid[i]
        phase = (q * model.crystal.get_asu_xyz(0)[0]).sum()
        f = q * torch.exp(1j * phase - 0.5 * q.square().sum() * model.ADP[0])
        fv = f @ model.V[i]
        weighted = fv.abs().square() * model.Winv[i].real
        expected.append(weighted.sum() if rank == -1 else weighted[rank])
    if not expected:
        assert actual.numel() == n_points
        return
    expected = torch.stack(expected)
    torch.testing.assert_close(actual[model.res_mask], expected)
    inputs = (model.q_grid, model.ADP, model.V, model.Winv)
    actual_grad = torch.autograd.grad(actual[model.res_mask].sum(), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), inputs)
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("arbitrary", [False, True])
@pytest.mark.parametrize("rank", [-1, 0, 2])
def test_excluded_modes_do_not_poison_intensity_gradients(arbitrary, rank):
    model = intensity_model("cpu", 5)
    with torch.no_grad():
        model.Winv[0, 0] = float('nan')
    if not arbitrary:
        model.use_arbitrary_q = False
        model.hsampling, model.ksampling, model.lsampling = [0, 0.8, 5], [0, 0, 1], [0, 0, 1]
        model.map_shape = (5, 1, 1)
    intensity = model.apply_disorder(rank=rank)
    if rank != 2:
        assert torch.isnan(intensity[0])
    loss = intensity[torch.isfinite(intensity)].sum()
    gradients = torch.autograd.grad(loss, (model.q_grid, model.ADP, model.V, model.Winv))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_arbitrary_q_setup_recovers_skew_cell_miller_indices(tmp_path):
    pdb = tmp_path / "skew.pdb"
    pdb.write_text("CRYST1   20.000   25.000   30.000  80.00 100.00 110.00 P 1           1\n"
                   "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\nEND\n")
    sampling = [-1, 1, 2]
    with patch.object(OnePhonon, "_setup_phonons"):
        grid = OnePhonon(str(pdb), sampling, sampling, sampling, device=torch.device("cpu"))
        arbitrary = OnePhonon(str(pdb), sampling, sampling, sampling,
                              q_vectors=grid.q_grid.detach(), device=torch.device("cpu"))
    torch.testing.assert_close(arbitrary.hkl_grid, grid.hkl_grid)
    torch.testing.assert_close(arbitrary.res_mask, grid.res_mask)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arbitrary", [False, True])
@pytest.mark.parametrize("use_data_adp", [False, True])
def test_protein_intensity_backward_is_finite(device, arbitrary, use_data_adp):
    pdb = Path(__file__).parent / "pdbs" / "5zck_p1.pdb"
    q = torch.tensor([[0.1, 0.2, 0.3], [0.15, 0.35, -0.4], [0, 0, 0]], dtype=torch.float64, device=device) if arbitrary else None
    model = OnePhonon(str(pdb), [-1, 1, 2], [-1, 1, 2], [-1, 1, 2],
                      q_vectors=q, device=torch.device(device))
    intensity = model.apply_disorder(use_data_adp=use_data_adp)
    finite = intensity[torch.isfinite(intensity)]
    assert finite.numel() > 0
    gradients = torch.autograd.grad(finite.sum(), (model.gamma_intra, model.gamma_inter, model.q_grid))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().max() > 0
