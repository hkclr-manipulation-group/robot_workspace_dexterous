# Spark2 V2 Dexterous Workspace

Compute RWS/DWS volumes, the DWS/RWS ratio, Jacobian manipulability, and
condition numbers with cuRobo.

## Model

```yaml
robot:
  urdf: ../cuarm_configuration/spark2_v2/share/no_gripper/arm_v2_stl.urdf
  base_link: base_link
  ee_links: [end_effector]
```

The program fits cuRobo collision spheres from the URDF collision geometry and
enables self-collision checking. The URDF, included Xacro files, and collision
meshes must be non-empty before running.

Spark2 V2 uses `package://arm_v2/meshes/arm/...` mesh URIs. Before fitting,
the program resolves these URIs to the repository's actual
`spark2_v2/share/meshes/arm` files and writes `output/normalized_robot.urdf`.

## Configuration

Edit `config.yaml`. Start with `resolution: 0.05` and
`orientations.count: 16`, then increase the resolution after validating the
model, link names, workspace bounds, and fitted collision spheres.

## Run

Run in a cuRobo V2 environment with CUDA-enabled PyTorch:

```bash
python run.py --config config.yaml --output-dir output --plot-height 0.30
```

## Outputs

- `<ee_link>.npz`: grid points, dexterity, manipulability, condition number,
  and minimum singular value.
- `<ee_link>_summary.json`: RWS/DWS volumes and the DWS/RWS ratio.
- `*.png`: dexterity, manipulability, and condition-number maps.

Strict DWS requires every sampled orientation to be reachable, represented by
`minimum_dexterity: 1.0`.
