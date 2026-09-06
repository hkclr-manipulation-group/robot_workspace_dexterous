# Spark2 V2 Dexterous Workspace

Compute reachable and dexterous workspace volumes, the DWS/RWS ratio, Jacobian
manipulability, and condition numbers with cuRobo.

## Standalone model inputs

All supplied configurations use `models/<model>/` inside this project:

```text
robot_workspace_dexterous/
  models/<model>/
    robot.urdf
    collision_spheres.yaml
    self_collision_ignore.yaml
  configs/<model>.yaml
  output/<model>/
```

No `cuarm_mesh`, `cuarm_configuration`, collision-generation repository, or STL
files are required. Copy this project with `models/` and `configs/` to the CUDA
machine. Python/cuRobo dependencies are still required.

The URDF contains link inertias, joints, origins, axes, limits and tool frames;
visual/collision geometry is removed. Collision checks use the external sphere
YAML. Runtime also strips visual/collision elements from user-supplied URDFs
without resolving mesh paths, writing `normalized_robot.urdf` under the output
directory. The source URDF is preserved. Plots use kinematics and collision
spheres, so mesh-free operation includes the dual-arm overview.

Existing presets retain their original kinematics, sampling settings and effective
self-collision ignore mappings. Spark2 v1 uses the actual URDF root
`dummy_base_link`; its fixed transform to `arm_base_link` is identity, so the
workspace coordinate frame does not change. The three design presets `D20260901B`,
`D20260902B60`, `D20260903B10` start with a 0.10 m grid, 16 orientations, 4 IK
seeds and adjacent-link collision exclusions. They retain exported joint ranges
[-3.14, 3.14] and velocity=0; validate those values against the design before
using results as hardware specifications. EE frames are Link06 / L6 / L6,
respectively, not independently calibrated TCPs.

The three design configs explicitly set
`robot.joint_limit_defaults: {velocity: 1.0, effort: 1.0}` for static workspace
evaluation. Missing or zero velocity/effort limits are replaced only in the
runtime `normalized_robot.urdf`, with a warning; positive limits and position
bounds are preserved. For these revolute joints the placeholders are 1 rad/s
and 1 N m, not measured hardware ratings. Source URDFs and collision bundles
retain the CAD values. Use actual limits for hardware or time-dependent work.
Without explicit defaults, `--validate-only` rejects nonpositive or nonfinite
dynamic limits and invalid position bounds before cuRobo initialization.
This prevents `lower velocity limits must be less than upper velocity limits`
(zero velocity gives [0, 0]) and the analogous zero-effort error.

CPU-only input validation:

```bash
python run.py --config configs/D20260901B.yaml --validate-only
```

Generate or update collision spheres in the collision project, then use its
`model_bundle export` command to copy the three compact files into the matching
model folder here. Update the local config's base and EE names if necessary.
Export is an explicit update step, not a runtime project dependency. When
updating existing models, preserve the intended joint limits and TCP transforms.
Review any non-adjacent collision exclusions; the migrated exclusions are
preserved settings, not newly validated physical collision rules.

## Installation

Use a Linux environment with CUDA-enabled PyTorch and cuRobo, then install this
project:

```bash
python3 -m pip install -e .
```

## Run

```bash
python3 run.py --config configs/spark2_v2.yaml --output-dir output/spark2_v2 --plot-height 0.0
```

Dual-arm presets are available under `configs/`. Select one by passing its
path, for example:

```bash
python3 run.py --config configs/dual_v2_1_no_gripper.yaml \
  --output-dir output/dual_v2_1_no_gripper
```

Each dual-arm preset uses `dummy_world` as the base and computes the left and
right workspaces from `arm_end_effector_l` and `arm_end_effector_r`.
The presets also load their generated `self_collision_ignore` YAML. External
ignore pairs are merged with any small robot-specific mapping written directly
in the dexterous config.

The program validates that every sphere link, end-effector link, base link, and
self-collision ignore link exists in the URDF before starting the GPU workload.

## Recommended workflow

Start with a coarse validation run:

```yaml
grid:
  resolution: 0.10
  z_step: 0.10
orientations:
  count: 16
solver:
  ik_seeds: 4
  batch_size: 4096
```

After checking bounds, link names, reachability, and collision behavior, use a
finer final run:

```yaml
grid:
  resolution: 0.05
  z_step: 0.05
orientations:
  count: 64
solver:
  ik_seeds: 8
  batch_size: 4096
```

### 120 GB GPU tuning

cuRobo already parallelizes IK on the GPU via `solver.batch_size`. With ~120 GB
VRAM, start at `4096` and increase until you hit OOM:

```yaml
solver:
  ik_seeds: 8
  batch_size: 4096   # try 8192 if stable
```

Runtime override without editing YAML:

```bash
python3 run.py --config configs/spark2_v2.yaml --batch-size 8192
```

If CUDA OOM appears, halve `batch_size`. Do not run multiple cuRobo processes on
the same GPU; one large batch is faster than several small ones.

Strict DWS requires every sampled orientation to be reachable and uses
`minimum_dexterity: 1.0`. Lower this threshold only when the task definition
allows a relaxed dexterous workspace.

Plots omit cells with zero reachable orientations instead of coloring the
entire unreachable grid. Manipulability and condition-number plots apply the
same reachability mask. At runtime the program reports the number of reachable
cells and maximum dexterity; it stops with a diagnostic error if every cell is
unreachable, and warns when only the selected DWS threshold is empty.

The dexterity image contains three orthogonal sections through the workspace:

- `XY` at `Z=0` by default; override it with `--plot-height`
- `XZ` at `Y=0`
- `YZ` at `X=0`

Plot axes default to the configured sampling bounds and can be overridden with
`plot.x_range`, `plot.y_range`, and `plot.z_range`. `plot.sections_xyz` selects
the YZ, XZ, and XY section coordinates respectively. Every view keeps an equal
metre scale, so robot-specific base heights are shown without clipping or
geometric distortion.

## Outputs

- `<ee_link>.npz`: grid points and all computed metrics
- `<ee_link>_filtered.npz`: points meeting the dexterity threshold
- `<ee_link>_summary.json`: RWS/DWS volumes and aggregate metrics
- `<ee_link>_dexterity_views.png`: XY/XZ/YZ dexterity center sections
- `<ee_link>_manipulability.png`: XY/XZ/YZ mean-manipulability sections
- `<ee_link>_condition_number.png`: XY/XZ/YZ worst-condition-number sections
- `normalized_robot.urdf`: mesh-free kinematic model used by cuRobo
- `dual_arm_workspace_overview.png`: left, right, and shared reachable
  workspaces together in one world-frame 3D view (multi-EE configs only)

Condition number is `sigma_max / sigma_min`: `1` is ideal and larger values are
worse. Its colorblind-friendlier blue-to-red plots use blue for values near `1`
and red for large or infinite values, without relying on green. The upper color
limit is the finite 95th percentile so isolated singularities do not compress
the useful color range.
