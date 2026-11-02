# MuJoCo contact-fidelity audit (2026-10-30), report only: nothing adopted
Scripts: mujoco_unit.py (unit tests, spec shared with the Drake spike), scene_contacts.py (real clamp contacts). MuJoCo 3.9.0.
Settings as the pan scene: dt 5 ms, elliptic cone, impratio 10, Newton, finger solref 0.005 / pan 0.02, solimp (0.9,0.95,0.001,0.5,2).

## Twist (two 20x20 mm pads at 10 N on a free block, mu 1; analytic uniform square patches: 0.153 Nm)
| setting | slip torque [Nm] | yaw at 1 s, before slip [deg] |
|---|---|---|
| condim 3, dt 5 ms (scene) | 0.090 | 0.80 |
| condim 3, dt 1 ms | 0.141 | 0.50 |
| condim 4/6, torsional = patch-matched 0.00765 m | 0.116 | 0.62 |
| condim 3 + noslip 10 | > 0.2 (no slip by test end; point-corner limit ~0.28) | 0.0005 |
Box-box contacts give 8 points (4 corners per pad). Soft contacts under-hold and creep; noslip over-holds (corner points).

## Creep (1 kg box on a plate, pushed at 0.5 mu m g; ideal Coulomb 0)
| setting | speed [mm/s] |
|---|---|
| scene (impratio 10), dt 5 or 1 ms | 0.128 |
| impratio 100 | 0.013 |
| noslip 10 | 0.00002 |

## Real clamp contacts (after the v1 clamp)
| world | left finger1/2 points (spread) | right finger1/2 points (spread) |
|---|---|---|
| I03 | 4 / 4 (24.8 mm) | 5 / 2 (14.3 mm) |
| V10 | 4 / 4 (24.8 mm) | 1 / 1 (0 mm) |
A single point contact with condim 3 has ZERO twist resistance: the pan is free to yaw about the right claw's jaw axis.
