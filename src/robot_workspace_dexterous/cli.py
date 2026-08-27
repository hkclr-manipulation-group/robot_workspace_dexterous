from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .curobo_solver import (
    build_collision_robots_from_config,
    build_collision_robots_from_urdf,
    compute_dexterous_workspace,
)
from .sampling import DexterousWorkspace
from .visualize import save_metric_top_view, save_top_view


def _progress(done: int, total: int, elapsed: float) -> None:
    if done == total or done <= 256 or done % (256 * 25) == 0:
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate if rate else float("inf")
        print(f"IK {done}/{total} ({100 * done / total:.1f}%), elapsed={elapsed/60:.1f} min, ETA={eta/60:.1f} min")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute a cuRobo dexterous workspace")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--plot-height", type=float, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    collision_robots = None
    if config.self_collision:
        if config.curobo_config_path is not None:
            print(f"loading existing collision model: {config.curobo_config_path}")
            collision_robots = build_collision_robots_from_config(
                str(config.curobo_config_path), config.base_link, config.ee_links
            )
        else:
            print("building cuRobo collision spheres and self-collision matrix from URDF...")
            collision_robots = build_collision_robots_from_urdf(
                str(config.urdf_path), config.base_link, config.ee_links,
                config.sphere_density, config.collision_matrix_samples,
                config.prune_collision_matrix,
                str(output_dir / "generated_collision_robot.yml"),
            )
    workspaces: dict[str, DexterousWorkspace] = {}
    for link in config.ee_links:
        print(f"computing {link}: {len(config.orientations)} orientations per XYZ cell")
        workspace = compute_dexterous_workspace(
            str(config.urdf_path or ""), config.base_link, link,
            config.x_range, config.y_range, config.heights, config.resolution,
            config.orientations, config.ik_seeds, config.batch_size,
            config.position_tolerance, config.orientation_tolerance,
            config.self_collision, _progress,
            None if collision_robots is None else collision_robots[link],
        )
        workspace.save(str(output_dir / f"{link}.npz"))
        filtered = workspace.threshold(config.minimum_dexterity)
        filtered.save(str(output_dir / f"{link}_filtered.npz"))
        plot = save_top_view(workspace, output_dir / f"{link}.png", link, args.plot_height)
        save_metric_top_view(
            workspace, workspace.manipulability_mean,
            output_dir / f"{link}_manipulability.png", f"{link} manipulability",
            "mean sqrt(det(J J^T))", args.plot_height,
        )
        save_metric_top_view(
            workspace, workspace.condition_number_max,
            output_dir / f"{link}_condition_number.png", f"{link} worst condition number",
            "max sigma_max / sigma_min (95th percentile color cap)", args.plot_height,
        )
        z_step = (
            float(np.min(np.diff(np.unique(config.heights))))
            if len(np.unique(config.heights)) > 1 else config.resolution
        )
        summary = workspace.volume_summary(
            config.minimum_dexterity, config.resolution ** 2 * z_step
        )
        finite_condition = workspace.condition_number_max[
            np.isfinite(workspace.condition_number_max)
        ]
        summary.update({
            "ee_link": link,
            "orientation_samples": workspace.orientation_count,
            "mean_manipulability_over_rws": float(np.mean(
                workspace.manipulability_mean[workspace.reachable_orientations > 0]
            )) if np.any(workspace.reachable_orientations > 0) else 0.0,
            "maximum_finite_condition_number": (
                float(np.max(finite_condition)) if len(finite_condition) else None
            ),
        })
        with (output_dir / f"{link}_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)
        workspaces[link] = workspace
        print(
            f"{link}: RWS={summary['rws_volume_m3']:.6f} m^3, "
            f"DWS={summary['dws_volume_m3']:.6f} m^3, "
            f"DWS/RWS={summary['dws_rws_ratio']:.4f}; plot={plot}"
        )
    if len(workspaces) > 1:
        ordered = [workspaces[link] for link in config.ee_links]
        shared = DexterousWorkspace(
            ordered[0].positions,
            np.minimum.reduce([item.dexterity for item in ordered]),
            np.minimum.reduce([item.reachable_orientations for item in ordered]),
            ordered[0].orientation_count,
        )
        shared.save(str(output_dir / "shared.npz"))
        save_top_view(shared, output_dir / "shared.png", "shared dexterous workspace", args.plot_height)


if __name__ == "__main__":
    main()
