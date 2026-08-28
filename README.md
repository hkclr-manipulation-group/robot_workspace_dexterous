# Spark2 V2 Dexterous Workspace

Compute reachable and dexterous workspace volumes, the DWS/RWS ratio, Jacobian
manipulability, and condition numbers with cuRobo.

## Model inputs

The project reads the robot model and precomputed collision spheres from sibling
repositories:

```text
workspace/
├── robot_workspace_dexterous/
├── cuarm_configuration/
│   └── spark2_v2/share/no_gripper/arm_v2_stl.urdf
└── collision_shpere_generation/
    └── examples/spark2_v2/collision_spheres.yaml
```

The collision spheres are loaded directly. Workspace computation does not fit
spheres, sample a collision matrix, or modify either source repository.

The default robot configuration is:

```yaml
robot:
  urdf: ../cuarm_configuration/spark2_v2/share/no_gripper/arm_v2_stl.urdf
  collision_spheres: ../collision_shpere_generation/examples/spark2_v2/collision_spheres.yaml
  base_link: arm_base_link
  ee_links: [arm_end_effector]
  self_collision_ignore:
    arm_base_link: [arm_L1]
    arm_L1: [arm_L2]
    arm_L2: [arm_L3]
    arm_L3: [arm_L4]
    arm_L4: [arm_L5]
    arm_L5: [arm_L6]
```

Adjacent links are ignored because their meshes overlap at physical joint
interfaces. Non-adjacent self-collisions remain enabled.

## Installation

Use a Linux environment with CUDA-enabled PyTorch and cuRobo, then install this
project:

```bash
python3 -m pip install -e .
```

## Run

```bash
python3 run.py --config config.yaml --output-dir output --plot-height 0.30
```

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
  batch_size: 512
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
  batch_size: 512
```

Strict DWS requires every sampled orientation to be reachable and uses
`minimum_dexterity: 1.0`. Lower this threshold only when the task definition
allows a relaxed dexterous workspace.

## Outputs

- `<ee_link>.npz`: grid points and all computed metrics
- `<ee_link>_filtered.npz`: points meeting the dexterity threshold
- `<ee_link>_summary.json`: RWS/DWS volumes and aggregate metrics
- `<ee_link>.png`: dexterity map
- `<ee_link>_manipulability.png`: mean manipulability map
- `<ee_link>_condition_number.png`: worst condition-number map
- `normalized_robot.urdf`: URI-normalized temporary model used by cuRobo
