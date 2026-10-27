"""Check derivatives of observables, not just the presence of grad_fn."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eryx.adapters import PDBToTensor
from eryx.models_torch import OnePhonon
from eryx.scatter_torch import structure_factors
from eryx.torch_utils import _SpectralMatrix
from tests.test_pytorch_audit import DEVICES, intensity_model


def spectral_model(matrix, pdos_mode=None):
    model = intensity_model(matrix.device, 3)
    model.n_cell = 1
    model.id_cell_ref = 0
    model.kvec = matrix.real.new_zeros(3, 3)
    model.Linv = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    model.gnm = SimpleNamespace()
    model.compute_hessian = lambda: matrix.reshape(1, 3, 1, 1, 3)
    if pdos_mode:
        model.pdos_mode = pdos_mode
        model.temperature_k = matrix.real.new_tensor(300.)
        model.pdos_omega = matrix.real.new_tensor([0., 1., 3., 5.]) * 1e14
        model.pdos_density = matrix.real.new_tensor([.5, 1., 2., 1.])
    model.compute_gnm_phonons()
    return model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("pdos_mode", [None, "direct", "thermal"])
@pytest.mark.parametrize("degenerate", [False, True])
def test_intensity_derivative_includes_mode_rotation(device, pdos_mode, degenerate):
    base = torch.diag(torch.tensor([2., 2. if degenerate else 5., 8.], dtype=torch.complex128, device=device))
    direction = base.new_tensor([[.1, .3 + .4j, -.2j], [.3 - .4j, -.3, .2], [.2j, .2, .5]])

    def observable(t):
        model = spectral_model(base + t * direction, pdos_mode)
        return model.apply_disorder()[model.res_mask]

    t = torch.tensor(0., dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(observable, (t,), eps=1e-5, atol=2e-7, rtol=2e-5)


def test_selected_rank_derivative_away_from_degeneracy():
    def observable(t):
        matrix = torch.diag(t.new_tensor([2., 5., 8.])).to(torch.complex128)
        direction = matrix.new_tensor([[0, 1j, .3], [-1j, 0, .2], [.3, .2, 0]])
        model = spectral_model(matrix + t * direction)
        return model.apply_disorder(rank=1)[model.res_mask]
    assert torch.autograd.gradcheck(observable, (torch.tensor(.2, dtype=torch.float64, requires_grad=True),))


def test_selected_rank_at_degeneracy_reports_undefined_derivative():
    matrix = (2 * torch.eye(3, dtype=torch.complex128)).requires_grad_()
    model = spectral_model(matrix)
    intensity = model.apply_disorder(rank=0)
    with pytest.raises(RuntimeError, match="degenerate"):
        intensity[model.res_mask].sum().backward()


def test_zero_population_does_not_hide_undefined_selected_rank_derivative():
    matrix = (2 * torch.eye(3, dtype=torch.complex128)).requires_grad_()
    model = spectral_model(matrix)
    model.pdos_mode = 'direct'
    model.pdos_omega = torch.tensor([1., 2.], dtype=torch.float64) * 1e14
    model.pdos_density = torch.tensor([1., 2.], dtype=torch.float64) - 2 ** .5
    model.compute_gnm_phonons()
    with pytest.raises(RuntimeError, match="degenerate"):
        model.apply_disorder(rank=0)[model.res_mask].sum().backward()


def test_inactive_degenerate_rank_does_not_poison_other_gradients():
    matrix = (2 * torch.eye(3, dtype=torch.complex128)).requires_grad_()
    model = spectral_model(matrix)
    (model._mode_covariance(0).real.sum() * 0).backward()
    torch.testing.assert_close(matrix.grad, torch.zeros_like(matrix))


@pytest.mark.parametrize("normalized", [False, True])
def test_population_edits_at_degeneracy(normalized):
    def observable(t):
        matrix = (2 * torch.eye(3, dtype=torch.complex128)) + t * torch.tensor(
            [[.1, .3j, 0], [-.3j, -.2, .4], [0, .4, .3]], dtype=torch.complex128)
        model = spectral_model(matrix)
        model.Winv = model.Winv / model.Winv.sum(dim=-1, keepdim=True) if normalized else model.Winv * 2
        return model.apply_disorder()[model.res_mask]
    assert torch.autograd.gradcheck(observable, (torch.tensor(0., dtype=torch.float64, requires_grad=True),))


def edited_covariance(matrix, edit):
    eigenvalues, vectors = torch.linalg.eigh(matrix)
    weights = edit(eigenvalues.reciprocal(), eigenvalues)
    return _SpectralMatrix.apply(matrix, eigenvalues, vectors.detach(), weights, None)


def test_global_population_normalization_derivative():
    def observable(t):
        matrix = torch.diag_embed(t.new_tensor([[2., 2.], [3., 5.]])).to(torch.complex128)
        direction = matrix.new_tensor([[[.1, .3j], [-.3j, -.2]], [[.2, .3], [.3, -.1]]])
        return edited_covariance(matrix + t * direction, lambda w, _: w / w.sum())
    assert torch.autograd.gradcheck(observable, (torch.tensor(0., dtype=torch.float64, requires_grad=True),))


def test_cross_q_population_edit_with_undefined_derivative_raises():
    matrix = torch.diag_embed(torch.tensor([[2., 2.], [3., 5.]], dtype=torch.complex128)).requires_grad_()
    def edit(weights, eigenvalues):
        weights = weights.clone()
        weights[1, 0] += eigenvalues[0, 0] - 2
        return weights
    covariance = edited_covariance(matrix, edit)
    with pytest.raises(RuntimeError, match="degenerate"):
        covariance[1, 0, 0].real.backward()


def test_cross_q_edit_cannot_split_a_degenerate_output_cluster():
    matrix = torch.diag_embed(torch.tensor([[2., 2.], [3., 5.]], dtype=torch.complex128)).requires_grad_()
    def edit(weights, eigenvalues):
        weights = weights.clone()
        delta = eigenvalues[1, 0] - 3
        weights[0] += torch.stack([delta, -delta])
        return weights
    covariance = edited_covariance(matrix, edit)
    with pytest.raises(RuntimeError, match="degenerate"):
        covariance[0, 0, 0].real.backward()


def test_higher_order_intensity_derivatives_fail_explicitly():
    t = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    model = spectral_model(torch.diag(t.new_tensor([2., 5., 8.])).to(torch.complex128) * t)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(model.apply_disorder()[model.res_mask].sum(), t, create_graph=True)


@pytest.mark.parametrize("pdos_mode", ["direct", "thermal"])
def test_population_and_temperature_derivatives(pdos_mode):
    def observable(density, temperature):
        model = spectral_model(torch.diag(torch.tensor([.015, .05, .12], dtype=torch.complex128)))
        model.pdos_mode = pdos_mode
        model.pdos_omega = torch.tensor([0., .1, .3, .6], dtype=torch.float64) * 1e14
        model.pdos_density = density
        model.temperature_k = temperature
        model.compute_gnm_phonons()
        return model.apply_disorder()[model.res_mask]
    density = torch.tensor([.5, 1., 2., 1.], dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor(300., dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(observable, (density, temperature), eps=1e-4, atol=1e-7)


def test_no_grad_intensity_does_not_create_trainable_output():
    with torch.no_grad():
        model = spectral_model(torch.eye(3, dtype=torch.complex128))
        assert not model.apply_disorder().requires_grad
        assert not model.Winv.requires_grad


def test_inference_mode_intensity():
    with torch.inference_mode():
        model = spectral_model(torch.eye(3, dtype=torch.complex128))
        assert torch.isfinite(model.apply_disorder()[model.res_mask]).all()


def test_adapter_preserves_existing_tensor_graph():
    source = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    adapter = PDBToTensor(device=torch.device('cpu'))
    converted = adapter.array_to_tensor(source.square(), dtype=torch.float64)
    gradient, = torch.autograd.grad(converted.sum(), source)
    torch.testing.assert_close(gradient, 2 * source)


@pytest.mark.parametrize("pdos_mode", ["direct", "thermal"])
def test_pdos_parameters_allow_repeated_optimization(tmp_path, pdos_mode):
    model = spectral_model(torch.diag(torch.tensor([.01, .04, .09], dtype=torch.complex128)))
    path = tmp_path / 'pdos.dat'
    np.savetxt(path, [[0., .5], [2., 1.], [6., 2.]])
    model.pdos_path = str(path)
    model.pdos_mode = pdos_mode
    model.temperature_k = torch.tensor(300., dtype=torch.float64, requires_grad=True)
    model._load_and_prepare_pdos()
    assert model.pdos_density.is_leaf
    optimizer = torch.optim.SGD([model.pdos_density, model.temperature_k], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        model.compute_gnm_phonons()
        loss = model.apply_disorder()[model.res_mask].sum()
        loss.backward()
        assert torch.isfinite(model.pdos_density.grad).all()
        if pdos_mode == 'thermal':
            assert torch.isfinite(model.temperature_k.grad)
        optimizer.step()


PDB = str(Path(__file__).parent / 'pdbs' / '5zck_p1.pdb')
SAMPLING = [-1, 1, 2]


@pytest.mark.parametrize("use_data_adp", [False, True])
def test_real_model_gamma_derivatives_match_finite_difference(use_data_adp):
    def observable(gamma, inter):
        model = OnePhonon(PDB, SAMPLING, SAMPLING, SAMPLING, gamma_intra=gamma,
                          gamma_inter=inter, device=torch.device('cpu'))
        intensity = model.apply_disorder(use_data_adp=use_data_adp)
        return intensity[torch.isfinite(intensity)].sum()
    gamma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    inter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(observable, (gamma, inter), eps=1e-4, atol=1e-3, rtol=1e-3)


def test_q_derivatives_include_phonons_and_independent_duplicate_points():
    def observable(q):
        model = OnePhonon(PDB, SAMPLING, SAMPLING, SAMPLING, q_vectors=q,
                          gamma_intra=1.3, gamma_inter=.7, device=torch.device('cpu'))
        return model.apply_disorder(use_data_adp=True)
    q = torch.tensor([[.11, .07, -.09], [.11, .07, -.09]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(observable, (q,), eps=1e-5, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_real_model_reuses_parameters_after_optimizer_step(dtype, device):
    gamma = torch.tensor(1.3, dtype=dtype, requires_grad=True)
    q = torch.tensor([[.11, .07, -.09]], dtype=dtype, requires_grad=True)
    model = OnePhonon(PDB, SAMPLING, SAMPLING, SAMPLING, q_vectors=q,
                      gamma_intra=gamma, gamma_inter=.7, device=torch.device(device))
    optimizer = torch.optim.SGD([gamma, q], lr=1e-8)
    previous = None
    for _ in range(2):
        optimizer.zero_grad()
        model.compute_gnm_phonons()
        model.compute_covariance_matrix()
        intensity = model.apply_disorder().sum()
        if previous is not None:
            assert intensity.item() != previous
        previous = intensity.item()
        intensity.backward()
        assert torch.isfinite(gamma.grad)
        assert torch.isfinite(q.grad).all()
        optimizer.step()
    torch.testing.assert_close(model.gamma_intra, gamma)


def test_structure_factors_gradcheck_all_continuous_inputs():
    torch.manual_seed(1)
    args = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
            for shape in [(2, 3), (2, 3), (2, 4), (2, 4), (2,), (2,), (6, 3)]]
    def observable(q, xyz, a, b, c, adp, projection):
        return structure_factors(q, xyz, a, b, c, U=adp, compute_qF=True,
                                 project_on_components=projection, sum_over_atoms=False)
    assert torch.autograd.gradcheck(observable, tuple(args))


def test_structure_factors_respect_no_grad():
    q = torch.ones(1, 3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        result = structure_factors(q, q, torch.ones(1, 4), torch.ones(1, 4), torch.ones(1))
    assert not result.requires_grad
