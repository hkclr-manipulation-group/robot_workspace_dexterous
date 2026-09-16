# Spark2 V2 Dexterous Workspace

Compute reachable and dexterous workspace volumes, the DWS/RWS ratio, Jacobian
manipulability, and condition numbers with cuRobo.

## Standalone model inputs

Collision exclusions were reviewed across all 11 bundles. Only the user-confirmed
`D20260901B` pair `Link04` / `Link05` remains ignored; other model lists are empty.
Old adjacency/zero-pose exclusions, including Spark2 v1/v2 `arm_L3` / `arm_L5`,
are archived in `models/ignore_rules_review.json` and are no longer active.
The active files match the collision viewer. This rule correction does not fix
gaps in the interior spheres: workspace IK remains a sphere approximation.
Removing exclusions can reduce reachable results, including intended joint
contacts; review actual geometry before approving further pair exceptions.

Runs save the first completed IK batch, then update partial results about every
30 seconds at batch boundaries and at completion. Open
`<output-dir>/progress/<ee_link>/index.html` in a browser for automatically
refreshing progress and XY/XZ/YZ projections. The timestamp identifies the latest
saved snapshot. Terminal progress, speed and ETA update about every 5 seconds
at batch boundaries; initialization or a long GPU batch must finish first.

Each progress folder contains `partial.npz`, `preview.png` and `status.json`.
The NPZ includes `tested_orientations` per cell and a `complete` flag. Unprocessed
cells are unknown; partial-cell dexterity is a lower bound until all orientations
are tested. Projections show all reachable cells without the final DWS filter,
using the maximum score across depth. Each file is replaced atomically. These
snapshots survive subsequent process failure but do not support automatic resume.

Preset grid spacing is now 25 mm for Spark2/Dual and 50 mm for D2026 models,
half the previous spacing on each axis (roughly eight times the IK work).
Sampling density is real; no interpolated points are added to reachability data.
Override XY and Z spacing together, or change the snapshot interval:

```bash
python -u run.py --config configs/spark2_v2.yaml --resolution 0.05 --snapshot-seconds 30 --output-dir output/spark2_v2
```

For `c10::AcceleratorError` / `an illegal memory access was encountered`, exit
the failed process and run a fresh diagnostic process (Linux):

```bash
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 python -u run.py --config configs/spark2_v2.yaml --debug-cuda --batch-size 32 --output-dir output/cuda_debug > dexterous_cuda.log 2>&1
```

Use the configuration that actually failed. Share `dexterous_cuda.log` and
`output/cuda_debug/cuda_debug.json`. Debug mode disables CUDA graphs, synchronizes
each major GPU stage, and records PyTorch/CUDA versions, GPU, loaded cuRobo path,
sphere set and batch settings. It preserves self-collision checks and all spheres.
The smaller batch is a diagnostic setting, not a confirmed fix for illegal memory
access. Normal execution retains CUDA graphs and the configured batch size.
This error still requires the first failing stage/stack to identify its cause;
an `unknown function` address by itself is insufficient.

All supplied configurations use `models/<model>/` inside this project:

```text
robot_workspace_dexterous/
  models/<model>/
    robot.urdf
    collision_spheres_interior.yaml
    collision_spheres_interior.json
    self_collision_ignore.yaml
  configs/<model>.yaml
  output/<model>/
```

No `cuarm_mesh`, `cuarm_configuration`, collision-generation repository, or STL
files are required. Copy this project with `models/` and `configs/` to the CUDA
machine. Python/cuRobo dependencies are still required.

All 11 presets now use the complete interior sphere sets, with **0 mm radius
expansion**. IK self-collision checks, CPU diagnostics and robot overview plots
read the same configured YAML. The adjacent JSON records generation details
and a checksum that is verified when loading. Original `collision_spheres.yaml`
files remain available; comparisons must use their matching URDF version.
The sphere domain is STL material only, preserving holes and cavities. Thin-wall
spheres may be very small. A complete set means every configured link is present,
not that its material volume or surface is completely covered. The reports and
workspace metadata retain this policy and label open-mesh containment estimates.
See [all interior files and previews](models/interior_gallery.html).

Interior spheres can leave uncovered regions, especially on open CAD meshes.
The reports label open-mesh estimates and measured surface gaps; collision-free
results refer to this sphere approximation. New workspace summaries identify the sphere file,
checksum, mode and whether self-collision checking was enabled. Existing files
under `output/` predate this update and must be recomputed to use interior spheres.

Four Dual presets previously used different kinematic revisions from the sphere
library: `dual_v2_1_left_hand_right_gripper`, `dual_v2_1_with_gripper`,
`dual_v2_2_left_hand_right_gripper_half`, and `dual_v2_2_no_gripper`. Their URDFs
now match the source models used for fitting, including wrist, gripper and trunk
frames. Their old standalone inputs and configuration are saved under each
model's `legacy/` directory. The migration originally retained and added some
pair exclusions. The subsequent ignore-rule review above supersedes those
exclusions; Dual 2.2 now has an empty active ignore list.
Its six zero-range trunk/head joints (S1–S4, H1, H2) are exported as fixed at
their existing zero positions, preserving link poses and avoiding invalid
zero-width joint limits in cuRobo.
See [migration details](models/interior_sync_report.json).

The URDF contains link inertias, joints, origins, axes, limits and tool frames;
visual/collision geometry is removed. Collision checks use the external sphere
YAML. Runtime also strips visual/collision elements from user-supplied URDFs
without resolving mesh paths, writing `normalized_robot.urdf` under the output
directory. The source URDF is preserved. Plots use kinematics and collision
spheres, so mesh-free operation includes the dual-arm overview.

The sampling density and ignore-rule updates above apply to all bundled models.
Apart from the four revisions above, model kinematics remain unchanged. Spark2 v1 uses the actual URDF root
`dummy_base_link`; its fixed transform to `arm_base_link` is identity, so the
workspace coordinate frame does not change. The three design presets `D20260901B`,
`D20260902B60`, `D20260903B10` use a 0.05 m grid, 16 orientations and 4 IK
seeds. Only D20260901B Link04/Link05 has an explicit exclusion. They retain exported joint ranges
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

For CPU joint-space collision diagnostics (2048 deterministic random poses):

```bash
python run.py --config configs/D20260901B.yaml --diagnose-only
# Short check before the default 2048-pose diagnostic:
python run.py --config configs/spark2_v2.yaml --diagnose-only --diagnostic-samples 32
```

This reports collision-free samples and per-link-pair collision frequencies; it
does not run IK and is not a proof of reachability. The CPU calculation blocks
both poses and sphere pairs to limit temporary memory with detailed sphere sets.
Old diagnostics from enclosing sphere sets do not describe the current interior
sets; rerun them after changing collision inputs. Inspect contacts against the
CAD model before changing collision exclusions.

The D20260903B10 J5 axis is tilted (`0 0.0078204 0.99997`). Runtime normalization
inserts an aligned joint frame and an inverse fixed transform for cuRobo's
cardinal-axis parser, retaining all original link poses and sphere coordinates.
Random-pose FK equivalence is regression-tested. This avoids the loader's
`str`/`value` failure without snapping the design axis to a different direction.

Generate or update interior spheres in the collision project with
`python -m collision_shpere_generation.interior_library`. Copy each model's
`<prefix>collision_spheres_interior.yaml` and matching JSON into its model folder
here, naming them `collision_spheres_interior.yaml` and `.json`. Keep YAML bytes
unchanged apart from LF/CRLF line-ending conversions, which are accepted by the
checksum validator. Then run `--validate-only`. A changed radius or center still
requires a matching generation report; an incomplete report is rejected separately.
This is an explicit update step, not a runtime project dependency. When
updating existing models, preserve the intended joint limits and TCP transforms.
Review every proposed collision exclusion, including adjacent pairs. Historical
exclusions are archived for reference and are not active rules.

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
  batch_size: 32
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
  batch_size: 32
```

### GPU memory budget and grid density

Bundled configurations use a 50 mm XYZ grid, batches of 32 IK targets, and
`solver.gpu_memory_fraction: 0.75`. On a 120 GB GPU the allocator budget is
at most about 90 GB, targeting a 30 GB reserve. The budget is further reduced
when other processes already occupy memory. This limits the PyTorch allocator;
external CUDA allocations and other processes are not capped by it.
The limit is applied before collision robots and IK solvers are constructed.

Spark and dual-arm grid spacing increased from 25 to 50 mm (about one eighth
as many XYZ cells); the D-series already used 50 mm. Orientations, IK seeds,
collision geometry and ignore rules remain unchanged. Coarser spacing reduces
spatial detail and total work, but batching controls the main per-batch GPU load.

```bash
python run.py --config configs/spark2_v2.yaml --resolution 0.05 --batch-size 32
```

Restart the process to apply these settings. An existing job keeps its old
allocations. If a batch still exceeds the budget, retry with `--batch-size 16`
or `8`; the allocator raises OOM rather than growing beyond its configured limit.

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
