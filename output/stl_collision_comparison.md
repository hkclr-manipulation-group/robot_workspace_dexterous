# Joint-contact policy comparison

32 deterministic joint samples per model; no IK grid was computed.

| Model | Strict: collision-free | Joint contacts allowed: collision-free |
| --- | ---: | ---: |
| D20260901B | 0/32 | 31/32 |
| D20260902B60 | 28/32 | 28/32 |
| D20260903B10 | 0/32 | 18/32 |
| dual_v2_1_left_hand_right_gripper | 0/32 | 1/32 |
| dual_v2_1_no_gripper | 0/32 | 30/32 |
| dual_v2_1_with_gripper | 0/32 | 0/32 |
| dual_v2_1_with_hand | 0/32 | 6/32 |
| dual_v2_2_left_hand_right_gripper_half | 0/32 | 2/32 |
| dual_v2_2_no_gripper | 0/32 | 32/32 |
| spark2_v1 | 0/32 | 20/32 |
| spark2_v2 | 0/32 | 21/32 |

The optional policy excludes whole link pairs within rigid assemblies and between bodies connected by one moving joint. All other pairs remain checked, subject to pre-existing exclusions.

Dual v2.1 with two grippers still has 0/32 passing samples: opposing left jaws collide in 30/32 samples and opposing right jaws in 28/32; no sampled pose passes all checks. This does not prove an empty workspace.

These are collision diagnostics, not full IK reachability or hardware validation.
