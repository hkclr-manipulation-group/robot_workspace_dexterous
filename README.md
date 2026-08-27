# Robot dexterous workspace

This package follows `robot_workspace_projection`, but computes a dexterous
workspace rather than boolean reachability. At each regular XYZ cell it asks
cuRobo collision-aware, multi-seed IK to solve a set of tool orientations:

`dexterity = number of reachable sampled orientations / number sampled`

The score is a discrete orientation-coverage index in `[0, 1]`. It is not a
Jacobian manipulability index. Self-collision is enabled by default, so a pose
counts only when cuRobo returns a collision-free IK solution.

The package also reports global equal-voxel volumes:

- RWS: cells with at least one reachable sampled orientation;
- DWS: RWS cells whose orientation coverage is at least
  `output.minimum_dexterity` (default `1.0`, meaning all sampled orientations);
- dexterous volume ratio: `DWS volume / RWS volume`.

For every successful IK solution, cuRobo evaluates the 6×N geometric Jacobian.
The NPZ stores the per-cell mean Yoshikawa index
`sqrt(det(J J^T)) = product(singular_values)`, its squared form `det(J J^T)`,
the worst condition number `sigma_max / sigma_min`, and the smallest observed
`sigma_min`. Infinite condition number means a numerically rank-deficient
solution. Linear and angular Jacobian rows have different physical units, so
absolute manipulability should be compared only for the same robot and frame.

## Run

Use a Python environment containing CUDA PyTorch and cuRobo V2:

```bash
cd robot_workspace_dexterous
cp config.example.yaml config.yaml
python run.py --config config.yaml --output-dir output --plot-height 0.30
```

The example reuses `../robot_workspace_projection/dual_arm.yaml` and its local
URDF. Paths inside the cuRobo YAML are resolved relative to that YAML file.

Start with 5 cm XYZ spacing and 16–32 orientations. A 1 cm grid with 128
orientations can contain millions of IK goals. After checking bounds and link
names, refine `resolution`, `z_step`, `orientations.count`, and `ik_seeds`.

Each arm produces:

- `<ee_link>.npz`: all grid positions, dexterity scores, successful orientation
  counts, and total orientation count;
- `<ee_link>_filtered.npz`: cells satisfying `minimum_dexterity`;
- `<ee_link>.png`: a top-view dexterity heat map (the nearest requested Z layer).
- `<ee_link>_manipulability.png` and `<ee_link>_condition_number.png`: Jacobian
  metric maps;
- `<ee_link>_summary.json`: RWS/DWS volumes, their ratio, and summary metrics.

For multiple arms, `shared.npz` and `shared.png` use the pointwise minimum arm
score. This means both arms independently cover at least that fraction of the
orientation samples; it does not claim simultaneous bimanual feasibility.

NPZ data can be loaded with:

```python
import numpy as np
data = np.load("output/arm_end_effector_l.npz")
positions = data["positions"]       # N x 3, metres in base_link
dexterity = data["dexterity"]       # N, range [0, 1]
manipulability = data["manipulability_mean"]
condition = data["condition_number_max"]
```
