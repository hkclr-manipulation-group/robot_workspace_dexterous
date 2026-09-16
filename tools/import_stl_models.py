"""Copy matching STL geometry into portable workspace model manifests."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from robot_workspace_dexterous.diagnostics import forward_kinematics


def import_models(source_root: Path) -> None:
    shared = PROJECT / "models" / "meshes"
    shared.mkdir(exist_ok=True)
    for config_path in sorted((PROJECT / "configs").glob("*.yaml")):
        model = config_path.stem
        source = source_root / model / "robot.urdf"
        destination = PROJECT / "models" / model
        urdf = destination / "robot.urdf"
        original, current = ET.parse(source).getroot(), ET.parse(urdf).getroot()
        rng = np.random.default_rng(42)
        q = {}
        for joint in original.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            low, high = ((-np.pi, np.pi) if joint.get("type") == "continuous" else
                         (float(limit.get("lower")), float(limit.get("upper"))))
            q[joint.get("name")] = rng.uniform(low, high, 8)
        old_poses, poses = forward_kinematics(original, q), forward_kinematics(current, q)
        links = {}
        for link in original.findall("link"):
            name = link.get("name")
            geometries = link.findall("collision") or link.findall("visual")
            entries = []
            for geometry in geometries:
                mesh = geometry.find("geometry/mesh")
                if mesh is None:
                    raise ValueError(f"{model}/{name}: non-STL geometry requires explicit conversion")
                if name not in poses or not np.allclose(old_poses[name], poses[name], atol=1e-8):
                    raise ValueError(f"{model}/{name}: source and workspace link frames differ")
                path = (source.parent / mesh.get("filename")).resolve()
                if path.suffix.lower() != ".stl":
                    raise ValueError(f"Expected STL: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                target = shared / f"{digest}.stl"
                if not target.exists():
                    shutil.copyfile(path, target)
                origin = geometry.find("origin")
                entries.append({
                    "file": "../meshes/" + target.name, "sha256": digest,
                    "scale": [float(v) for v in mesh.get("scale", "1 1 1").split()],
                    "xyz": [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()],
                    "rpy": [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()],
                })
            if entries:
                links[name] = entries
        manifest = {"version": 1, "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                    "links": links}
        (destination / "collision_meshes.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["robot"]["collision_meshes"] = f"../models/{model}/collision_meshes.yaml"
        config["solver"]["collision_backend"] = "stl"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"{model}: {len(links)} STL collision links; link-frame equivalence checked", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=PROJECT.parent / "collision_shpere_generation" / "models")
    import_models(parser.parse_args().source_root)
