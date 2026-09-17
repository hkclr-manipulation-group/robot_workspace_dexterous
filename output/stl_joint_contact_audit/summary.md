# STL collision audit

Deterministic joint samples and STL collision only; no IK or workspace grid was run.

Existing ignores retained; each model records the selected joint-contact policy. Persistent sampled contact is not proof of unavoidable collision or permission to ignore a pair.

| Model | Collision-free samples | Pairs colliding in every pair sample |
| --- | ---: | --- |
| D20260901B | 31/32 | None |
| D20260902B60 | 28/32 | None |
| D20260903B10 | 18/32 | None |
| dual_v2_1_left_hand_right_gripper | 1/32 | None |
| dual_v2_1_no_gripper | 30/32 | None |
| dual_v2_1_with_gripper | 0/32 | None |
| dual_v2_1_with_hand | 6/32 | None |
| dual_v2_2_left_hand_right_gripper_half | 2/32 | None |
| dual_v2_2_no_gripper | 32/32 | None |
| spark2_v1 | 20/32 | None |
| spark2_v2 | 21/32 | None |
