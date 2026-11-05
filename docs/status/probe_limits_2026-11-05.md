# What the two NERO arms can probe (2026-11-05)

> Status (2026-11-05): CURRENT (11-05). Every row is conditional on the ASSUMED joint-torque read error; open questions at the end.

USER 11-05: "with the limits of our arm what is possible to be probed first, let us establish that and note down".
Simulation and model only. No hardware.

Numbers: `world_model/v2/nero/contents/probe_limits.py` (MODEL, linear map from joint-torque read error to wrench error at the
probe pose). Raw output: `/data/local/ss5772/contents/probe_limits_2026-11-05/{LIMITS.json,TABLE.md}`.
Check: the model mass error at 0.05 Nm (0.0783 kg) equals the probe's own recorded noise bound (0.0784-0.0791 kg, 274 rungs).

## Sensing this rests on
- Per arm: joint encoders + a per-joint torque read (commanded torque or motor current). No wrist F/T sensor.
- **Whether the real NERO reports a usable joint torque, and its error, is UNKNOWN.** Every row below is conditional on it.
- Read error per joint, both ASSUMED: 0.05 Nm (`probe.TAU_NOISE`, friction known or dithered out) and 0.30 Nm (the sim
  frictionloss, `config.ASSUMED`; the static-friction band if friction is not modelled).
- Probe pose: the grasp (rest) pose on the counter. It is the same capture for box and trays, so the numbers are the same for
  every task.

## Result (MODEL; worst case over the 14 joints)

| quantity | how | error at 0.05 Nm | error at 0.30 Nm | range / limit | probeable? |
|---|---|---|---|---|---|
| total mass M | static hold, object off the counter | 0.078 kg | 0.47 kg | weighable up to **2.96 kg total (box), 3.13 kg (trays)**; binds = per-claw payload rating 3 kg - gripper at u 0.9, worst CoM (torque row 0.78) | yes, up to ~3 kg; above it only "M >= ~2.9 kg" (a lower bound: refuse without lifting) |
| CoM, horizontal (x / y) | moment of the static hold | 19 / 32 mm at 1 kg; 6 / 11 mm at 3 kg | 116 / 193 mm at 1 kg; 39 / 64 mm at 3 kg | same as mass | yes if friction is modelled; no otherwise |
| CoM height | needs a tilt phi | 614 mm (3 deg), 185 mm (10 deg) at 1 kg | 3.7 m / 1.1 m | tilt capped by R (3 deg) / U (10 deg) thresholds, spill | **no** |
| moving contents (slosh / sliding) | force change m*a during a move | smallest detectable moving mass 3.4 kg at 0.5 m/s^2 | 20 kg | a capped by A_NOM 0.5 m/s^2 (ASSUMED); arm-inertia model error not counted | **no** (by force reads) |
| per-mug fill (tray) | static weigh gives 3 numbers (M, M c_x, M c_y) | - | - | unknown dimensions: tray2 0 of 2, tray4 1 of 4, tray6 3 of 6 | tray2 yes (noise permitting); tray4/6 no |

Prior totals vs weighable max: box 0.75-5.25 kg (part above 2.96 kg: lower bound only); tray2 0.90-1.40; tray4 1.50-2.50;
tray6 2.10-3.60 (part above 3.13 kg: lower bound only).

Not covered here: the lift-off ("free") test. Today it fails above ~2 kg (11-05 variant runs). Any weigh needs a reliable "off the
counter" signal first.

## What published probing skills report (full texts read 11-05; [S] = from a model summary of the page, not verbatim)

| paper | robot / sensors | object start | motion | identifies | reported accuracy |
|---|---|---|---|---|---|
| Zhang, Zhou, Vasudevan 2504.21486 (UMich, RSS 2025; code github.com/roahmlab/OnlineSafeSysID) | Kinova Gen3 7-DoF, joint torque sensors [S] | in hand | 4 safe exciting trajectories, 7.5 s [S]; joint/torque limits + obstacles kept | 10 inertial params, guaranteed interval | interval [1.2, 5.2] kg around 3.2 kg nominal (Table I) |
| Lao, Han, Ma, Chirikjian 2104.09858 (NUS/JHU) | 4-DoF OpenMANIPULATOR-X, encoders only | in hand | static holds on a pose grid; learned torque model with friction term sgn(w) | mass, CoM | mass MAE 3.7 g (4.75 %), CoM 8.8 mm, objects 50-150 g |
| Shan, Pham 2409.03369 (NTU) | Denso VS060 6-DoF, motor current | in hand | 4 s trajectory, 3 deg/s with reversals | mass, CoM | test payloads 0.68-1.21 kg; errors not stated |
| Jin, Mo, Yuan 2502.02663 (UIUC; U-GRAPH) | UR5e + wrist F/T 1 kHz | in hand | 2 wrist orientations | CoM only | 1.47 cm, 7.6 %; objects 43-613 g |

None of the four starts with the object resting on a support, and none identifies moving contents.

## Open (user)
1. Target demo. Which hidden parameters the demo uses (mass, horizontal CoM, per-mug fill, contents motion) and which object
   priors it uses, given the table above.
2. NERO joint torque read: does it exist, and what are its error and friction? Without a friction model the errors move to the
   0.30 Nm row.
3. Lift-off detection, which comes before any weigh (option still open from the 11-05 variant runs).
