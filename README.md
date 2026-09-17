# Dexterous Workspace

Compute reachable and dexterous workspace volumes, DWS/RWS, Jacobian
manipulability and condition numbers with cuRobo. All 11 supplied configurations
now use **STL self-collision detection** by default.

## STL collision checking

The checker loads original STL triangles once and builds persistent local-frame
BVHs on the GPU. For each batch it computes full-robot forward kinematics,
rejects disjoint link bounding boxes, then tests candidate triangles through
BVHs. Closed meshes also receive containment checks, including disconnected
components. Contact tolerance is 1 micrometre. Geometry is not simplified or
replaced by spheres; scale and geometry origins come from the source URDF.

Open CAD meshes have no unambiguous solid interior. They are checked for surface
intersection and are listed in `open_mesh_links` in validation and results.
Holes are not filled. Explicit `self_collision_ignore` rules still apply;
adjacent links are not automatically ignored. Only the existing D20260901B
Link04/Link05 exception is present in the bundled presets. Designed joint
contacts can therefore make all sampled configurations collide. Inspect the
reported pair frequencies before deciding whether an exception is appropriate.

cuRobo generates pose-IK candidates without sphere collision costs. **Every
returned seed** is checked against STL, and a goal counts as reachable if at
least one pose-valid candidate passes. Jacobian metrics use that selected
candidate. STL does not provide an optimization gradient: a finite seed set
can miss a collision-free alternative. Increasing `solver.ik_seeds` expands the
search at additional cost. This is sampled reachability, not a proof that an
unaccepted target has no feasible solution.

STL avoids gaps in a sphere approximation, but is not guaranteed to be faster.
Triangle count, overlapping link bounds and the number of valid IK seeds affect
runtime. Diagnostics measure FK plus collision checking, not full IK or
end-to-end workspace speed. First use includes Warp kernel compilation.

## Install and run

Use the existing CUDA-enabled PyTorch/cuRobo v2 environment and update this
project. STL mode automatically supports v2 loaders that lack the workspace's
`kinematic_link_names` extension; reinstalling cuRobo is not required for that
specific compatibility error.

```bash
python -m pip install -e .
python -u run.py --config configs/spark2_v2.yaml --output-dir output/spark2_v2_stl
```

Older v2 loaders print `STL compatibility mode`. This path retains every robot
chain through the existing collision-link loader and allocates two disabled
placeholder spheres to avoid legacy empty-result handling. Sphere collision
costs stay disabled, and all feasibility decisions still use the STL checker.
The updated sibling cuRobo can instead use the sphere-free path; install it with
`python -m pip install -e ../curobo --no-build-isolation` if desired. The optional
sphere broad-phase backend still requires that updated checkout.

Warp 1.6 or newer is required for GPU STL checking; cuRobo supplies Warp in its
runtime environment. Restart existing processes after updating. Normal IK runs
retain CUDA graphs. A full workspace run requires CUDA-enabled PyTorch.

Portable input layout:

```text
models/meshes/<sha256>.stl
models/<model>/robot.urdf
models/<model>/collision_meshes.yaml
models/<model>/self_collision_ignore.yaml
configs/<model>.yaml
```

Copy the whole `models/` and `configs/` directories. STL assets are shared by
content hash (90 unique files, approximately 130 MB). Runtime does not require
the collision-generation repository. The manifest checks URDF and STL
checksums and supplies each mesh's scale and origin. The geometry-free URDF is
used for kinematics; `normalized_robot.urdf` is generated for cuRobo without
changing STL coordinates. To reimport assets after an intentional model
update, run `python tools/import_stl_models.py` in a workspace containing the
sibling `collision_shpere_generation` project. The importer verifies FK
equivalence before copying geometry.

CPU-only validation and a short GPU collision diagnostic:

```bash
python run.py --config configs/spark2_v2.yaml --validate-only
python run.py --config configs/spark2_v2.yaml --diagnose-only --diagnostic-samples 32
```

Diagnostics use deterministic random joint configurations and report the number
passing self-collision, throughput, and per-pair frequencies on at most 32
samples. The separately timed throughput excludes the per-pair diagnostic.
These runs do not solve IK. STL validation loads geometry but needs neither
PyTorch nor a GPU.

For a fresh CUDA error diagnostic (Linux):

```bash
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 python -u run.py --config configs/spark2_v2.yaml --debug-cuda --batch-size 8 --output-dir output/cuda_debug
```

Debug mode disables IK CUDA graphs and synchronizes major stages. Normal runs
keep graphs enabled. Use a new process after an illegal-memory-access error.

## Sampling, models and progress

Bundled grids use 50 mm spacing, batch size 32 and a PyTorch memory fraction of
0.75. Warp allocations and other processes are outside the PyTorch allocator
cap. Spark/Dual presets use 32 orientations and 8 seeds; the three D2026 presets
use 16 orientations and 4 seeds. CLI overrides include:

```bash
python -u run.py --config configs/dual_v2_2_no_gripper.yaml --resolution 0.05 --batch-size 16 --snapshot-seconds 30
```

Dual presets compute both tool workspaces in the `dummy_world` frame, retaining
both arms and all other moving chains for STL checks. Zero-range joints in the
Dual 2.2 exports remain fixed. Arbitrary joint axes are normalized for cuRobo
without changing FK. D2026 presets explicitly provide static-evaluation defaults
for missing/zero effort and velocity limits; those are placeholders, not hardware
ratings. Source joint positions and limits remain unchanged.

The first completed batch is saved, then partial results update approximately
every 30 seconds at batch boundaries. Open
`<output-dir>/progress/<ee_link>/index.html` for an automatically refreshing
preview. Its `partial.npz`, `preview.png` and `status.json` survive a later failure
but do not implement resume. Unprocessed cells are unknown, and partial-cell
dexterity is a lower bound until every orientation is tested. Terminal speed
and ETA update at batch boundaries.

Strict DWS uses `minimum_dexterity: 1.0`: every sampled orientation must pass.
Final plots omit unreachable cells and show XY/XZ/YZ sections with equal metre
scales. If every cell is unreachable, the run reports a diagnostic error.
Previous sphere-based results in `output/` must be recomputed for STL semantics.

### No reachable grid cells

This error means no sampled cell has even one accepted IK orientation. Lowering
`minimum_dexterity` cannot restore rejected targets; that threshold only filters
the accepted results afterward.

For Spark2 v2, inspect the STL collision pairs first:

```bash
python run.py --config configs/spark2_v2.yaml --diagnose-only --diagnostic-samples 32
```

In a local 32-pose diagnostic, `arm_L1`/`arm_L2` and `arm_L5`/`arm_L6`
collided in every sample, with zero collision-free configurations. They are
directly connected by `arm_J2` and `arm_J6`; the supplied ignore list is empty.
This explains a collision rejection mechanism independently of workspace bounds.
Random joint sampling does not prove that every IK solution collides.

Review whether these contacts are intended joint interfaces or geometry/model
errors. If the model owner confirms permitted contacts, add only the reviewed
pairs to `models/spark2_v2/self_collision_ignore.yaml`; otherwise correct the
collision geometry. Keep all other pairs checked. Then rerun the diagnostic
before the full grid. A collision-free joint sample is not proof of target IK
reachability. The default configuration does not add exclusions automatically.

## Outputs

- `<ee_link>.npz`: sampled positions, reachability and Jacobian metrics.
- `<ee_link>_filtered.npz`: cells meeting the dexterity threshold.
- `<ee_link>_summary.json`: volumes, metrics and collision-model metadata.
- `<ee_link>_dexterity_views.png`, `_manipulability.png`, `_condition_number.png`:
  orthogonal sections.
- `normalized_robot.urdf`: kinematics used by cuRobo.
- `dual_arm_workspace_overview.png`: multi-tool workspaces in a shared frame.

## Optional sphere backend

The old sphere inputs remain available for explicit comparison:

```bash
python run.py --config configs/spark2_v2.yaml --collision-backend spheres
```

This uses complete interior sphere sets and the GPU link bounding-sphere
broad phase; it remains a sphere approximation. `--no-collision-broad-phase`
selects the legacy exhaustive backend, whose sphere-index and shared-memory
limits can prevent large models from running. The sphere-only microbenchmark
`python benchmark_collision.py --config configs/spark2_v2.yaml` compares the new
sphere kernel with and without culling. It is not an STL benchmark or an
end-to-end speed comparison. Existing interior-generation reports and historical
ignore-rule reviews remain under `models/` for reference.
