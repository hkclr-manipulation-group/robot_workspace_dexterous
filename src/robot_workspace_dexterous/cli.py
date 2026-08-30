from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .curobo_solver import (
    build_collision_robots,
    compute_dexterous_workspace,
)
from .sampling import DexterousWorkspace
from .visualize import (
    load_zero_pose_collision_spheres,
    save_dexterity_center_views,
    save_dual_workspace_overview,
    save_metric_center_views,
)


def _progress(done: int, total: int, elapsed: float) -> None:
    if done == total or done <= 256 or done % (256 * 25) == 0:
        rate = done / elapsed if elapsed else 0.0
        eta = (total - done) / rate if rate else float("inf")
        print(f"IK {done}/{total} ({100 * done / total:.1f}%), elapsed={elapsed/60:.1f} min, ETA={eta/60:.1f} min")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute a cuRobo dexterous workspace")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--plot-height",
        type=float,
        default=None,
        help="Override the configured Z coordinate for the XY section",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override solver.batch_size (use 4096–8192 on ~120 GB GPUs)",
    )
    parser.add_argument(
        "--ik-seeds",
        type=int,
        default=None,
        help="Override solver.ik_seeds",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.batch_size is not None or args.ik_seeds is not None:
        from dataclasses import replace
        config = replace(
            config,
            batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
            ik_seeds=args.ik_seeds if args.ik_seeds is not None else config.ik_seeds,
        )
    plot_sections = (
        config.plot_sections
        if args.plot_height is None
        else (config.plot_sections[0], config.plot_sections[1], args.plot_height)
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    collision_robots = None
    if config.self_collision:
        print(f"loading collision spheres: {config.collision_spheres_path}")
        collision_robots = build_collision_robots(
            str(config.urdf_path),
            str(config.collision_spheres_path),
            config.base_link,
            config.ee_links,
            config.self_collision_ignore,
            str(output_dir / "normalized_robot.urdf"),
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
        reachable_cells = int(np.count_nonzero(workspace.reachable_orientations > 0))
        if reachable_cells == 0:
            raise RuntimeError(
                f"{link}: no reachable grid cells were found; check workspace bounds, "
                "base/tool link names, and self-collision ignores before plotting"
            )
        print(
            f"{link}: {reachable_cells}/{len(workspace.positions)} grid cells reachable, "
            f"maximum dexterity={float(np.max(workspace.dexterity)):.4f}"
        )
        workspace.save(str(output_dir / f"{link}.npz"))
        filtered = workspace.threshold(config.minimum_dexterity)
        if len(filtered.positions) == 0:
            print(
                f"warning: no cells meet minimum_dexterity="
                f"{config.minimum_dexterity:.4f}; unfiltered RWS outputs are still saved"
            )
        filtered.save(str(output_dir / f"{link}_filtered.npz"))
        plot = save_dexterity_center_views(
            workspace,
            output_dir / f"{link}_dexterity_views.png",
            link,
            sections_xyz=plot_sections,
            axis_ranges=config.plot_ranges,
            base_position=config.base_position,
        )
        save_metric_center_views(
            workspace, workspace.manipulability_mean,
            output_dir / f"{link}_manipulability.png", f"{link} manipulability",
            "mean sqrt(det(J J^T))",
            sections_xyz=plot_sections,
            axis_ranges=config.plot_ranges,
            base_position=config.base_position,
        )
        save_metric_center_views(
            workspace, workspace.condition_number_max,
            output_dir / f"{link}_condition_number.png", f"{link} worst condition number",
            "max sigma_max / sigma_min (95th percentile color cap)",
            sections_xyz=plot_sections,
            axis_ranges=config.plot_ranges,
            base_position=config.base_position,
            cmap="coolwarm",
            vmin=1.0,
            cap_positive_infinity=True,
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
        overview = save_dual_workspace_overview(
            workspaces,
            output_dir / "dual_arm_workspace_overview.png",
            config.plot_ranges,
            config.base_position,
            robot_spheres=load_zero_pose_collision_spheres(
                config.urdf_path, config.collision_spheres_path
            ),
        )
        print(f"dual-arm workspace overview: {overview}")
        ordered = [workspaces[link] for link in config.ee_links]
        shared = DexterousWorkspace(
            ordered[0].positions,
            np.minimum.reduce([item.dexterity for item in ordered]),
            np.minimum.reduce([item.reachable_orientations for item in ordered]),
            ordered[0].orientation_count,
        )
        shared.save(str(output_dir / "shared.npz"))
        save_dexterity_center_views(
            shared,
            output_dir / "shared_dexterity_views.png",
            "shared dexterous workspace",
            sections_xyz=plot_sections,
            axis_ranges=config.plot_ranges,
            base_position=config.base_position,
        )


if __name__ == "__main__":
    main()
