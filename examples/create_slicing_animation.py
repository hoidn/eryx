"""Run after examples.pytorch_diffuse: python -m examples.create_slicing_animation.

Requires imageio and pillow. Planes and axes use voxel indices.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from eryx.visualization import create_slicing_animation

SOURCE = Path("tmp/pytorch-quickstart/results.npz")
OUTPUT = SOURCE.with_name("slicing.gif")
N_FRAMES = 12


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError("Run python -m examples.pytorch_diffuse first")
    create_slicing_animation(
        str(SOURCE), normal=[1, 1, 1], n_frames=N_FRAMES,
        output_path=str(OUTPUT), nan_strategy="mask", fps=6,
        subsample=2, figsize=(6, 5), dpi=80,
    )
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
