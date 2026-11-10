"""Plot the uniform-limit review slice from compare.py's JSON output."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    points = np.asarray(data["targets_m"])
    xs, xi = np.unique(points[:, 0], return_inverse=True)
    zs, zi = np.unique(points[:, 2], return_inverse=True)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), layout="constrained", sharex=True, sharey=True)
    for ax, row in zip(axes.flat, data["models"]):
        grid = np.full((len(zs), len(xs)), np.nan)
        grid[zi, xi] = [len(ids) for ids in row["accepted_orientation_ids"]]
        image = ax.pcolormesh(xs, zs, grid, shading="nearest", cmap="viridis",
                              vmin=0, vmax=data["orientations"])
        ax.set(title=row["model"], aspect="equal", xticks=xs[::2], yticks=zs[::2])
    axes.flat[-1].set_visible(False)
    fig.supxlabel("Base X (m)")
    fig.supylabel("Base Z (m)")
    fig.suptitle("Native bare flange at Y = 0 m  |  J1–J6: ±180°")
    fig.colorbar(image, ax=list(axes.flat[:-1]), ticks=[0, 4, 8, 12, 16],
                 label="Accepted orientations out of 16", shrink=.75)
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
