# ASSUMED-value sweep, task box

Every swept value stays **ASSUMED (break-even, not measured)**. A break-even is the value that would be needed, not an accepted one. Baseline = today's values.

## cell 1.0-1.5
baseline: ok=False, failing rows: joint_pos (margin -0.03106187975596164), rule1[2:s2_place:stored_backup] (margin None), obj_tilt (margin 6.085100000000001)

| knob | today | target rows (margin) | break-even | change | next failing row(s) at break-even | justify (measurement that would make it MEASURED) |
|---|---|---|---|---|---|---|
| A_BACKUP | 0.25 | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the retrace acceleration the real arms + gripper hold the load at without slip (measured slip / torque margin on hardware) |
| A_MIN | 0.05 | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | a task-timing choice (the slowest acceptable carry); no measurement, a user ruling |
| BIN_WIDTH_KG | 0.5 | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | not a physical value: the reading-bin design choice; the finest bin the claw-load read can resolve = its measured noise (EPS_FREE + nb) |
| CLEARANCE_MIN | 5 mm | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the pose / calibration error of the arm links vs the world (hand-eye + FK error), measured on the robot, plus the model-vs-mesh distance error |
| CLEAR_M_BOX | 0.03 | joint_pos, rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the measured object-pose error at the lift / carry (perception + tracking) over the shelf edge |
| DY_RANGE | -0.04..0.06 | joint_pos | none on grid | | target rows fail | the free lateral space at the counter / shelf (measured scene layout) bounding how far the lift may be shifted sideways |
| EPS_CC_POS | 2 mm | - | None | | knob's rows pass today; the baseline fails on other rows | the stiffness / compliance of the real weld grasp + the arms' FK error: the object-pose disagreement the real gripper pair can absorb (measured internal force vs offset) |
| EPS_CC_ROT | 0.01 | - | None | | knob's rows pass today; the baseline fails on other rows | same as EPS_CC_POS (rotation disagreement the weld absorbs) |
| MARGIN_WANT | 0.05 | joint_pos | none on grid | | target rows fail | same as PATH_MARGIN_MIN (J7 tracking error): the planner target should be the checker margin |
| PATH_MARGIN_BOTH | 0.05 | joint_pos (-0.0311) | 0.015 rad (linear estimate of the zero crossing 0.018938, not a run) | -0.035 rad | rule1[2:s2_place:stored_backup] (None); obj_tilt (6.0851) | J7 / joint-limit tracking error (one number for the checker margin and the planner target) |
| PATH_MARGIN_MIN | 0.05 | joint_pos (-0.0311) | 0.015 rad (linear estimate of the zero crossing 0.018938, not a run) | -0.035 rad | rule1[2:s2_place:stored_backup] (None); obj_tilt (6.0851) | J7 / joint-limit tracking error of the executed path vs the planned path (measured max over episodes), plus the URDF limit vs the firmware soft limit |
| PSI_STEP_DEG | 3 | joint_pos | none on grid | | target rows fail | not a physical value: a planner search resolution (finer = more IK arm-angle candidates); no measurement needed, only run time |
| TAU_NOISE | 0.05 | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | NERO joint-torque read noise: measure the read at rest / constant hold on the real arm (sd and max over n samples per joint) |
| U_OBJ_CLEAR_M | 5 mm | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the object pose + shape error vs the world geometry (perception / calibration), measured on the robot |

## cell 2.5-3.0
baseline: ok=False, failing rows: a_le_a_admit (margin -0.4127), joint_pos (margin -0.03106187975596164), rule1[0:s0_lift:stored_backup] (margin None), rule1[1:s1_move:stored_backup] (margin None), rule1[2:s2_place:stored_backup] (margin None), obj_tilt (margin 4.2257)

| knob | today | target rows (margin) | break-even | change | next failing row(s) at break-even | justify (measurement that would make it MEASURED) |
|---|---|---|---|---|---|---|
| A_BACKUP | 0.25 | rule1[0:s0_lift:stored_backup], rule1[1:s1_move:stored_backup], rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the retrace acceleration the real arms + gripper hold the load at without slip (measured slip / torque margin on hardware) |
| A_MIN | 0.05 | rule1[0:s0_lift:stored_backup], rule1[1:s1_move:stored_backup], rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | a task-timing choice (the slowest acceptable carry); no measurement, a user ruling |
| BIN_WIDTH_KG | 0.5 | rule1[0:s0_lift:stored_backup], rule1[1:s1_move:stored_backup], rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | not a physical value: the reading-bin design choice; the finest bin the claw-load read can resolve = its measured noise (EPS_FREE + nb) |
| CLEARANCE_MIN | 5 mm | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the pose / calibration error of the arm links vs the world (hand-eye + FK error), measured on the robot, plus the model-vs-mesh distance error |
| CLEAR_M_BOX | 0.03 | joint_pos, rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the measured object-pose error at the lift / carry (perception + tracking) over the shelf edge |
| DY_RANGE | -0.04..0.06 | joint_pos | none on grid | | target rows fail | the free lateral space at the counter / shelf (measured scene layout) bounding how far the lift may be shifted sideways |
| EPS_CC_POS | 2 mm | - | None | | knob's rows pass today; the baseline fails on other rows | the stiffness / compliance of the real weld grasp + the arms' FK error: the object-pose disagreement the real gripper pair can absorb (measured internal force vs offset) |
| EPS_CC_ROT | 0.01 | - | None | | knob's rows pass today; the baseline fails on other rows | same as EPS_CC_POS (rotation disagreement the weld absorbs) |
| MARGIN_WANT | 0.05 | joint_pos | none on grid | | target rows fail | same as PATH_MARGIN_MIN (J7 tracking error): the planner target should be the checker margin |
| PATH_MARGIN_BOTH | 0.05 | joint_pos (-0.0311) | 0.015 rad (linear estimate of the zero crossing 0.018938, not a run) | -0.035 rad | a_le_a_admit (-0.4127); rule1[0:s0_lift:stored_backup] (None); rule1[1:s1_move:stored_backup] (None); rule1[2:s2_place:stored_backup] (None); obj_tilt (4.2257) | J7 / joint-limit tracking error (one number for the checker margin and the planner target) |
| PATH_MARGIN_MIN | 0.05 | joint_pos (-0.0311) | 0.015 rad (linear estimate of the zero crossing 0.018938, not a run) | -0.035 rad | a_le_a_admit (-0.4127); rule1[0:s0_lift:stored_backup] (None); rule1[1:s1_move:stored_backup] (None); rule1[2:s2_place:stored_backup] (None); obj_tilt (4.2257) | J7 / joint-limit tracking error of the executed path vs the planned path (measured max over episodes), plus the URDF limit vs the firmware soft limit |
| PSI_STEP_DEG | 3 | joint_pos | none on grid | | target rows fail | not a physical value: a planner search resolution (finer = more IK arm-angle candidates); no measurement needed, only run time |
| TAU_NOISE | 0.05 | rule1[0:s0_lift:stored_backup], rule1[1:s1_move:stored_backup], rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | NERO joint-torque read noise: measure the read at rest / constant hold on the real arm (sd and max over n samples per joint) |
| U_OBJ_CLEAR_M | 5 mm | rule1[2:s2_place:stored_backup] | none on grid | | target rows fail | the object pose + shape error vs the world geometry (perception / calibration), measured on the robot |

## not swept

- rule1.K: not swept: no consumer. rule1.K and rule1.NODE_CAP are only RECORDED (rule1.py:153-154 assumptions, rule1_contents.py:621/643 `worst`, rule4.py:81); no search reads them (the R_K witness is the depth-1 planner admission). Raising them cannot change a verdict.
- rule1.NODE_CAP: not swept: no consumer (see rule1.K)
- rule4.MAX_ROUNDS / RETRACE_HALVINGS: not requested here; rule4_contents module globals, add to KNOBS if a Rule 4 near-miss needs them
