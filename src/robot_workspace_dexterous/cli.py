from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import load_config
from .cuda_debug import enable_cuda_debug, cuda_environment, cuda_stage
from .curobo_solver import (
    build_collision_robots,
    collision_sphere_metadata,
    validate_robot_inputs,
    _normalized_urdf_for_curobo,
    _prepare_joint_limits,
    compute_dexterous_workspace,
)
from .sampling import DexterousWorkspace
from .progress import WorkspaceProgress
from .visualize import (
    load_zero_pose_collision_spheres,
    save_dexterity_center_views,
    save_dual_workspace_overview,
    save_metric_center_views,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute a cuRobo dexterous workspace")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "configs/spark2_v2.yaml"))
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
        help="Override solver.batch_size (default auto-scales from 32 using available GPU memory)",
    )
    parser.add_argument(
        "--ik-seeds",
        type=int,
        default=None,
        help="Override solver.ik_seeds",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate URDF and collision YAML on CPU, without running IK")
    parser.add_argument("--diagnose-only", action="store_true", help="Sample joint configurations: GPU STL or CPU sphere collision diagnostics")
    parser.add_argument("--diagnostic-samples", type=int, default=2048, help="Joint configurations for --diagnose-only")
    parser.add_argument('--resolution', type=float, help='Override XY and Z grid spacing in metres')
    parser.add_argument('--snapshot-seconds', type=float, default=30, help='Save partial results after this interval at batch boundaries (default 30 s)')
    parser.add_argument("--debug-cuda", action="store_true",
                        help="Synchronize CUDA stages, disable CUDA graphs and save environment details; default batch size 32")
    parser.add_argument("--no-collision-broad-phase", action="store_true",
                        help="Use legacy exhaustive GPU self-collision for comparison (may exceed memory limits)")
    parser.add_argument("--collision-backend", choices=["stl", "spheres"],
                        help="Override the configured collision representation (bundled models: stl)")
    parser.add_argument("--allow-joint-contacts", action="store_true",
                        help="Deprecated alias for the default adjacent-joint policy")
    parser.add_argument("--strict-collision", action="store_true",
                        help="Check adjacent joint-interface geometry too (audit mode; may reject all IK seeds)")
    args = parser.parse_args(argv)
    if args.resolution is not None and (not np.isfinite(args.resolution) or args.resolution <= 0):
        parser.error('--resolution must be finite and positive')
    if not np.isfinite(args.snapshot_seconds) or args.snapshot_seconds <= 0:
        parser.error('--snapshot-seconds must be finite and positive')
    if args.debug_cuda:
        enable_cuda_debug()
        if args.batch_size is None:
            args.batch_size = 32
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.ik_seeds is not None and args.ik_seeds < 1:
        parser.error('--ik-seeds must be positive')
    config = load_config(args.config)
    if args.collision_backend is not None:
        from dataclasses import replace
        config = replace(config, collision_backend=args.collision_backend)
    if args.allow_joint_contacts and config.collision_backend != "stl":
        parser.error("--allow-joint-contacts requires the STL collision backend")
    contact_ignores = config.self_collision_ignore
    if not args.strict_collision:
        from .joint_contacts import merge_contact_ignores
        contact_ignores = merge_contact_ignores(str(config.urdf_path), contact_ignores)
    if args.resolution is not None:
        from dataclasses import replace
        config = replace(config, resolution=args.resolution,
                         heights=np.arange(float(config.heights[0]), float(config.heights[-1]) + args.resolution * .01, args.resolution))
    mesh_model = None
    if config.collision_backend == "stl":
        from .mesh_model import load_mesh_model
        if config.collision_meshes_path is None:
            parser.error("STL checking requires robot.collision_meshes")
        root = ET.parse(config.urdf_path).getroot()
        _prepare_joint_limits(root, config.joint_limit_defaults)
        links = {link.get("name") for link in root.findall("link")}
        children = {child.get("link") for child in root.findall("joint/child")}
        if config.base_link not in links - children or set(config.ee_links) - links:
            raise ValueError("Invalid URDF root or end-effector for STL checking")
        mesh_model = load_mesh_model(config.collision_meshes_path, config.urdf_path,
                                     contact_ignores)
        if not args.strict_collision:
            from .joint_contacts import allow_joint_contacts
            mesh_model = allow_joint_contacts(mesh_model, config.urdf_path)
            mesh_model.metadata["joint_contact_excluded_pairs"] = [
                list(pair) for pair in sorted({
                    tuple(sorted((first, second)))
                    for first, others in contact_ignores.items() for second in others
                })
            ]
            print("Adjacent-joint contact policy enabled; excluded interface pairs: " +
                  json.dumps(mesh_model.metadata["joint_contact_excluded_pairs"]), flush=True)
        collision_model = mesh_model.metadata
        if collision_model.get("open_mesh_links"):
            print("WARNING: open STL links use surface-only collision; hollow-space occupancy is unknown: "
                  + ", ".join(collision_model["open_mesh_links"]), flush=True)
    else:
        if config.collision_spheres_path is None:
            parser.error("Sphere checking requires robot.collision_spheres")
        validate_robot_inputs(str(config.urdf_path), str(config.collision_spheres_path),
                              config.base_link, config.ee_links, contact_ignores,
                              config.joint_limit_defaults)
        collision_model = {
            **collision_sphere_metadata(config.collision_spheres_path),
            "joint_contact_policy": "strict" if args.strict_collision else "adjacent_joint_interfaces",
            "joint_contact_excluded_pairs": [] if args.strict_collision else [list(pair) for pair in sorted(
                {tuple(sorted((first, second))) for first, others in contact_ignores.items() for second in others}
            )],
        }
    if args.diagnose_only:
        if mesh_model is not None:
            from .mesh_collision import diagnose_mesh_collisions
            result = diagnose_mesh_collisions(mesh_model, config.urdf_path, args.diagnostic_samples)
        else:
            from .diagnostics import diagnose_collisions
            result = diagnose_collisions(config.urdf_path, config.collision_spheres_path,
                                         contact_ignores, samples=args.diagnostic_samples)
        print(json.dumps(result, indent=2))
        return
    if args.validate_only:
        print(f"Validated {args.config}: {config.base_link} -> {', '.join(config.ee_links)}; collision mode={collision_model['mode']}")
        if mesh_model is not None:
            print(json.dumps(collision_model, indent=2))
        return
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
    monitors = {link: WorkspaceProgress(output_dir / 'progress' / link, link,
                {'collision_model': collision_model, 'resolution_m': config.resolution,
                 'self_collision_enabled': config.self_collision,
                 'self_collision_broad_phase': config.self_collision and config.collision_backend == 'spheres' and not args.no_collision_broad_phase})
                for link in config.ee_links}
    for monitor in monitors.values():
        print(f'Live progress: {monitor.directory / "index.html"}', flush=True)
    if args.debug_cuda:
        details = {**cuda_environment(), 'config': str(Path(args.config).resolve()),
                   'collision_model': collision_model, 'batch_size': config.batch_size,
                   'ik_seeds': config.ik_seeds, 'self_collision': config.self_collision,
                   'self_collision_broad_phase': config.self_collision and config.collision_backend == 'spheres' and not args.no_collision_broad_phase,
                   'use_cuda_graph': False}
        (output_dir / 'cuda_debug.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
        print(json.dumps(details, indent=2), flush=True)
    normalized_urdf = _normalized_urdf_for_curobo(
        str(config.urdf_path), str(output_dir / "normalized_robot.urdf"), config.joint_limit_defaults
    )
    collision_robots = None
    from .gpu_memory import configure_gpu_memory, recommend_batch_size
    allocation_limit = configure_gpu_memory(config.gpu_memory_fraction)
    if args.batch_size is None:
        from dataclasses import replace
        selected_batch = recommend_batch_size(config.batch_size, allocation_limit)
        if selected_batch != config.batch_size:
            config = replace(config, batch_size=selected_batch)
            print(f'Auto batch size: {selected_batch} (allocation limit {allocation_limit/2**30:.1f} GiB)', flush=True)
    mesh_checker = None
    if config.self_collision and mesh_model is not None:
        from .mesh_collision import MeshCollisionChecker
        import torch
        print(f"STL self-collision: {collision_model['total_triangles']:,} triangles; "
              f"checking all {config.ik_seeds} IK candidates per goal", flush=True)
        if collision_model['open_mesh_links']:
            print("Open STL links use surface-intersection checks: " +
                  ", ".join(collision_model['open_mesh_links']), flush=True)
        with cuda_stage('build STL BVHs', args.debug_cuda):
            mesh_checker = MeshCollisionChecker(mesh_model, config.urdf_path,
                                                 device=f'cuda:{torch.cuda.current_device()}')
    elif config.self_collision:
        print(f"loading collision spheres: {config.collision_spheres_path}", flush=True)
        print("Self-collision: " + ("legacy exhaustive sphere pairs" if args.no_collision_broad_phase
                                  else "GPU link bounding spheres + internal sphere refinement"), flush=True)
        with cuda_stage('build collision robots', args.debug_cuda):
            collision_robots = build_collision_robots(
                normalized_urdf,
                str(config.collision_spheres_path),
                config.base_link,
                config.ee_links,
                contact_ignores,
                self_collision_broad_phase=not args.no_collision_broad_phase,
            )
    workspaces: dict[str, DexterousWorkspace] = {}
    for link in config.ee_links:
        print(f"computing {link}: {len(config.orientations)} orientations per XYZ cell; grid {config.resolution*1000:g} mm", flush=True)
        workspace = compute_dexterous_workspace(
            normalized_urdf, config.base_link, link,
            config.x_range, config.y_range, config.heights, config.resolution,
            config.orientations, config.ik_seeds, config.batch_size,
            config.position_tolerance, config.orientation_tolerance,
            config.self_collision, monitors[link].progress,
            None if collision_robots is None else collision_robots[link],
            debug_cuda=args.debug_cuda,
            snapshot=monitors[link].snapshot,
            snapshot_seconds=args.snapshot_seconds,
            mesh_checker=mesh_checker,
        )
        reachable_cells = int(np.count_nonzero(workspace.reachable_orientations > 0))
        if reachable_cells == 0:
            raise RuntimeError(
                f"{link}: no reachable grid cells were found; check workspace bounds, "
                "base/tool link names, and self-collision ignores before plotting. "
                "This means zero accepted IK targets, not just zero cells meeting minimum_dexterity. "
                "To inspect collision rejection, run: "
                f'python run.py --config "{args.config}" --collision-backend {config.collision_backend} '
                + ("--allow-joint-contacts " if args.allow_joint_contacts else "") +
                "--diagnose-only --diagnostic-samples 32. "
                "Persistent contacts between joint housings may require model-specific review; "
                "collision pairs are not ignored automatically."
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
            "collision_model": {**collision_model, "self_collision_enabled": config.self_collision,
                                "self_collision_broad_phase": config.self_collision and config.collision_backend == 'spheres' and not args.no_collision_broad_phase},
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
            robot_spheres=None if config.collision_backend == 'stl' else load_zero_pose_collision_spheres(
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
