"""Run from the repository root: python -m examples.pytorch_diffuse."""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from eryx.models_torch import OnePhonon
from eryx.visuals import visualize_central_slices

PDB = "tests/pdbs/histidine.pdb"
SAMPLING = (-2, 2, 3)
DEVICE = torch.device("cpu")
OUT = Path("tmp/pytorch-quickstart")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    model = OnePhonon(
        PDB, hsampling=SAMPLING, ksampling=SAMPLING, lsampling=SAMPLING,
        model="gnm", group_by="asu", expand_p1=True, res_limit=0.0,
        gnm_cutoff=4.0, gamma_intra=1.0, gamma_inter=1.0,
        device=DEVICE, n_processes=1,
    )
    intensity = model.apply_disorder(use_data_adp=True)
    I = intensity.detach().cpu().numpy()
    q = model.q_grid.detach().cpu().numpy()
    hkl = model.hkl_grid.detach().cpu().numpy()
    volume = I.reshape(model.map_shape)
    valid = np.isfinite(I)
    assert I.size == np.prod(model.map_shape)
    assert valid.any() and np.max(I[valid]) > 0

    np.savez_compressed(
        OUT / "results.npz", intensity=I, q_vectors=q, hkl_grid=hkl,
        map_shape=model.map_shape,
        adp=model.ADP.detach().cpu().numpy(),
        covariance=model.covar.detach().cpu().numpy(),
    )
    visualize_central_slices(volume, output=str(OUT / "slices.png"))
    plt.close("all")

    qnorm = np.linalg.norm(q, axis=1)
    profile, edges, _ = binned_statistic(qnorm[valid], I[valid],
                                        statistic="mean", bins=20)
    pdos = model.generate_pdos(bins=30)
    np.savetxt(OUT / "pdos.dat", pdos, header="frequency_THz_empirical density")
    np.savetxt(OUT / "radial.dat", np.column_stack(((edges[1:]+edges[:-1])/2, profile)),
               header="q_inverse_angstrom mean_intensity")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot((edges[1:]+edges[:-1])/2, profile)
    axes[0].set(xlabel="|q| (1/Angstrom)", ylabel="Mean diffuse intensity")
    axes[1].plot(pdos[:, 0], pdos[:, 1])
    axes[1].set(xlabel="Frequency (THz; empirical scale)", ylabel="Mode density")
    fig.tight_layout()
    fig.savefig(OUT / "diagnostics.png", dpi=150)
    plt.close(fig)
    print("Grid:", model.map_shape, "finite:", valid.sum(), "/", I.size)
    print("Intensity min/mean/max:", I[valid].min(), I[valid].mean(), I[valid].max())
    print("Outputs:", OUT.resolve())


if __name__ == "__main__":
    main()
