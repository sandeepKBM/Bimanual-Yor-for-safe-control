E1 accuracy (n per level; errors: 0)
| noise | n | fit ok | pos p50/p95/max mm | yaw |res| p50/p95/max deg | within 5 mm & 2 deg | within 5 mm | within 2 deg |
|---|---|---|---|---|---|---|---|
| clean | 200 | 200 | 0.02/0.04/0.06 | 0.06/0.21/0.40 | 200/200 | 200 | 200 |
| low | 200 | 200 | 0.03/0.09/0.13 | 0.06/0.21/0.39 | 200/200 | 200 | 200 |
| mid | 200 | 200 | 0.43/0.91/1.32 | 0.46/1.57/4.24 | 198/200 | 200 | 198 |
| zed | 200 | 200 | 1.08/2.30/3.38 | 1.30/4.68/9.42 | 137/200 | 200 | 137 |
| zed_calib | 200 | 200 | 3.27/6.24/8.06 | 1.43/5.07/8.81 | 105/200 | 156 | 131 |

E2 closed loop, plan pinch 0.65 / push 0.07, FULL prereg box (V = 32 vertices, I = random interior)
| noise | vertices held | interior held | refused (no fit) | failure categories | failed runs: residual mm / deg |
|---|---|---|---|---|---|
| clean | 20/32 | 27/30 | 0 | {'no_clean_pregrasp': 9, 'right_claw_lost_or_missed': 2, 'sag_or_no_lift': 4} | I06:nan/+nan I09:0.0/+0.0 I20:0.0/+0.0 V01:0.0/+0.1 V03:0.0/+0.1 V09:0.0/+0.1 V11:0.0/+0.1 V16:nan/+nan V17:nan/+nan V18:nan/+nan V19:nan/+nan V24:nan/+nan |
| mid | 20/32 | 27/30 | 0 | {'no_clean_pregrasp': 9, 'right_claw_lost_or_missed': 3, 'sag_or_no_lift': 3} | I06:nan/+nan I09:0.6/-0.1 I20:0.3/+0.9 V01:0.5/+0.4 V03:0.4/-0.9 V09:0.5/+1.0 V11:0.1/-1.4 V16:nan/+nan V17:nan/+nan V18:nan/+nan V19:nan/+nan V24:nan/+nan |
| zed | 19/32 | 27/30 | 0 | {'no_clean_pregrasp': 11, 'right_claw_lost_or_missed': 3, 'slip_or_tilt': 1, 'sag_or_no_lift': 1} | I06:nan/+nan I09:1.6/-0.2 I20:0.6/+2.8 V01:1.2/+0.9 V03:nan/+nan V09:1.2/+2.1 V10:0.9/+0.6 V11:nan/+nan V16:nan/+nan V17:nan/+nan V18:nan/+nan V19:nan/+nan |
| zed_calib | 19/32 | 25/30 | 0 | {'no_clean_pregrasp': 10, 'right_claw_lost_or_missed': 3, 'slip_or_tilt': 1, 'sag_or_no_lift': 3, 'still_on_support': 1} | I06:nan/+nan I09:2.6/+0.0 I15:2.9/-8.1 I17:4.1/-5.8 I20:1.8/+1.4 V01:3.4/+1.0 V03:nan/+nan V09:5.1/+1.8 V10:1.5/+1.3 V11:0.2/-2.2 V16:nan/+nan V17:nan/+nan |
