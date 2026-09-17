"""Audit STL self-collision across presets without running IK or changing ignores."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from robot_workspace_dexterous.config import load_config
from robot_workspace_dexterous.mesh_model import load_mesh_model


def classify_pairs(result: dict, urdf: Path) -> None:
    """Describe structural relationships without declaring contacts permissible."""
    root = ET.parse(urdf).getroot()
    joints = {}
    for joint in root.findall("joint"):
        pair = frozenset((joint.find("parent").get("link"), joint.find("child").get("link")))
        joints[pair] = {"name": joint.get("name"), "type": joint.get("type")}
    for pair in result["pair_frequencies"]:
        pair["direct_joint"] = joints.get(frozenset(pair["links"]))
        pair["all_pair_samples_collide"] = pair["colliding_samples"] == result["pair_diagnostic_samples"]


def write_report(output: Path, results: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Deterministic joint samples and STL collision only; no IK or workspace grid was run.",
        "policy": "Existing ignores retained. Persistent sampled contact is not proof of unavoidable collision or permission to ignore a pair.",
        "models": results,
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    lines = ["# STL collision audit", "", report["scope"], "", report["policy"], "",
             "| Model | Collision-free samples | Pairs colliding in every pair sample |",
             "| --- | ---: | --- |"]
    for name, result in results.items():
        if "error" in result:
            lines.append(f"| {name} | Error | {result['error'].replace('|', '/').replace(chr(10), ' ')} |")
            continue
        persistent = [p for p in result["pair_frequencies"] if p["all_pair_samples_collide"]]
        descriptions = []
        for pair in persistent:
            relationship = f"joint {pair['direct_joint']['name']}" if pair["direct_joint"] else "not directly adjacent"
            descriptions.append(f"{' / '.join(pair['links'])} ({relationship})")
        lines.append(f"| {name} | {result['collision_free_samples']}/{result['samples']} | {'; '.join(descriptions) or 'None'} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path,
                        help="Explicit config files; defaults to all bundled configs")
    parser.add_argument("--samples", type=int, default=32,
                        help="Joint samples per model; pair frequencies use at most 32")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "output/stl_collision_audit")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    from robot_workspace_dexterous.mesh_collision import diagnose_mesh_collisions
    configs = args.configs or sorted((PROJECT / "configs").glob("*.yaml"))
    if len({p.stem for p in configs}) != len(configs):
        parser.error("config filenames must have distinct stems")
    results = {}
    for config_path in configs:
        name = config_path.stem
        print(f"Auditing {name}: {args.samples} joint samples", flush=True)
        try:
            config = load_config(config_path)
            if config.collision_meshes_path is None:
                raise ValueError("This audit requires an STL mesh manifest")
            model = load_mesh_model(config.collision_meshes_path, config.urdf_path,
                                    config.self_collision_ignore)
            result = diagnose_mesh_collisions(model, config.urdf_path, args.samples)
            classify_pairs(result, config.urdf_path)
            result["config"] = str(config_path.resolve())
            result["self_collision_ignore"] = config.self_collision_ignore
            persistent = [p for p in result["pair_frequencies"] if p["all_pair_samples_collide"]]
            print(f"{name}: {result['collision_free_samples']}/{args.samples} collision-free; "
                  f"{len(persistent)} persistent sampled pairs", flush=True)
            del model
        except Exception as exc:
            result = {"config": str(config_path.resolve()), "error": f"{type(exc).__name__}: {exc}"}
            print(f"{name}: {result['error']}", flush=True)
        results[name] = result
        write_report(args.output_dir, results)
        gc.collect()
    print(f"Report: {args.output_dir.resolve() / 'summary.md'}", flush=True)
    return int(any("error" in result for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
