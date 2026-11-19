# Depth box fit: error table (sampled)

Guarantee: **sampled** over the drawn worlds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] x seeds [0, 1, 2, 3] (250 runs).  Cameras: ['room_front', 'room_left', 'room_back', 'room_right']; image size [2208, 1242] (w, h).
Truth = geom poses read from the model.  Errors in metres, yaw in degrees.  "max over n" = max |error| over the n observed fits (a measured maximum, not a bound).

Noise model (zed2i_lab): {"sigma_z": "z^2*SIGMA_D_PX/(f_px*BASELINE_M)", "SIGMA_D_PX": 0.2, "BASELINE_M": 0.12, "LAB_CORR_FRAC": 0.5, "LAB_CORR_PX_720": 15, "LAB_DROP": 0.05, "LAB_CALIB_M": 0.003, "LAB_CALIB_DEG": 0.1, "status": "ASSUMED (user approved 11-19) except BASELINE_M = datasheet"}
Clean: no noise, deterministic, seed 0 only.

## box_bookcase / clean

| entity | field | n obs | n unobs | p50 | p95 | max over n | max + | max - | worst world |
|---|---|---|---|---|---|---|---|---|---|
| crate | cx | 50 | 0 | 2.438e-06 | 5.886e-06 | max over n = 50: 6.659e-06 | 5.934e-06 | -6.659e-06 | w43 s0 |
| crate | cy | 50 | 0 | 1.028e-05 | 3.554e-05 | max over n = 50: 5.046e-05 | 4.013e-05 | -5.046e-05 | w43 s0 |
| crate | cz | 50 | 0 | 0.0003006 | 0.0005331 | max over n = 50: 0.0005662 | 0.0005662 | 0 | w42 s0 |
| crate | sx | 50 | 0 | 1.24e-05 | 1.774e-05 | max over n = 50: 2.067e-05 | 1.443e-06 | -2.067e-05 | w7 s0 |
| crate | sy | 50 | 0 | 6.3e-05 | 0.0001004 | max over n = 50: 0.000148 | 0 | -0.000148 | w43 s0 |
| crate | sz | 50 | 0 | 0.0006414 | 0.001081 | max over n = 50: 0.001162 | 0 | -0.001162 | w42 s0 |
| crate | yaw | 50 | 0 | 0.00495 | 0.01103 | max over n = 50: 0.01149 | 0.01114 | -0.01149 | w35 s0 |
| crate | top_z | 50 | 0 | 1.36e-05 | 2.288e-05 | max over n = 50: 2.522e-05 | 8.69e-06 | -2.522e-05 | w22 s0 |
| counter | cx | 50 | 0 | 1.321e-05 | 2.562e-05 | max over n = 50: 4.123e-05 | 4.123e-05 | -6.898e-06 | w26 s0 |
| counter | cy | 36 | 14 | 8.911e-06 | 2.888e-05 | max over n = 36: 3.121e-05 | 2.858e-05 | -3.121e-05 | w19 s0 |
| counter | cz | 0 | 50 | - | - | - | - | - | - |
| counter | sx | 50 | 0 | 1.638e-05 | 3.745e-05 | max over n = 50: 4.773e-05 | 4.773e-05 | -4.586e-05 | w13 s0 |
| counter | sy | 36 | 14 | 4.59e-05 | 8.877e-05 | max over n = 36: 0.0001549 | 0 | -0.0001549 | w43 s0 |
| counter | sz | 0 | 50 | - | - | - | - | - | - |
| counter | yaw | 50 | 0 | 0.003821 | 0.008615 | max over n = 50: 0.009157 | 0.006884 | -0.009157 | w20 s0 |
| counter | top_z | 50 | 0 | 0.0001045 | 0.0001518 | max over n = 50: 0.0002006 | 0 | -0.0002006 | w29 s0 |
| bookcase_l1_floor | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | sy | 50 | 0 | 3.191e-05 | 3.949e-05 | max over n = 50: 5.207e-05 | 0 | -5.207e-05 | w27 s0 |
| bookcase_l1_floor | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | yaw | 50 | 0 | 0.001242 | 0.00266 | max over n = 50: 0.003436 | 0.003116 | -0.003436 | w3 s0 |
| bookcase_l1_floor | top_z | 50 | 0 | 5.26e-05 | 5.583e-05 | max over n = 50: 6.304e-05 | 0 | -6.304e-05 | w41 s0 |
| bookcase_l1_base | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | sy | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | yaw | 50 | 0 | 0.0001237 | 0.0004037 | max over n = 50: 0.0005321 | 0.0005321 | -0.0003958 | w9 s0 |
| bookcase_l1_base | top_z | 50 | 0 | 2.894e-05 | 3.028e-05 | max over n = 50: 3.058e-05 | 0 | -3.058e-05 | w24 s0 |
| bookcase_plinth | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | sy | 50 | 0 | 3.084e-05 | 3.467e-05 | max over n = 50: 3.714e-05 | 0 | -3.714e-05 | w27 s0 |
| bookcase_plinth | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | yaw | 50 | 0 | 0.000259 | 0.001126 | max over n = 50: 0.001555 | 0.001555 | -0.001495 | w15 s0 |
| bookcase_plinth | top_z | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | sy | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | yaw | 50 | 0 | 5.743e-05 | 0.0001693 | max over n = 50: 0.00036 | 0.00036 | -0.0002268 | w5 s0 |
| bookcase_mid | top_z | 50 | 0 | 3.109e-05 | 3.246e-05 | max over n = 50: 3.297e-05 | 0 | -3.297e-05 | w19 s0 |
| bookcase_top | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | sy | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | yaw | 50 | 0 | 2.813e-05 | 9.245e-05 | max over n = 50: 0.0001585 | 8.826e-05 | -0.0001585 | w2 s0 |
| bookcase_top | top_z | 50 | 0 | 1.44e-05 | 1.575e-05 | max over n = 50: 1.611e-05 | 0 | -1.611e-05 | w24 s0 |
| bookcase_side_left | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | sy | 50 | 0 | 5.628e-05 | 0.0001217 | max over n = 50: 0.000135 | 0 | -0.000135 | w27 s0 |
| bookcase_side_left | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | yaw | 50 | 0 | 0.01264 | 0.01575 | max over n = 50: 0.01642 | 0 | -0.01642 | w24 s0 |
| bookcase_side_left | top_z | 1 | 49 | 0.0003594 | 0.0003594 | max over n = 1: 0.0003594 | 0 | -0.0003594 | w35 s0 |
| bookcase_side_right | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | sy | 50 | 0 | 5.391e-05 | 7.241e-05 | max over n = 50: 7.315e-05 | 0 | -7.315e-05 | w17 s0 |
| bookcase_side_right | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | yaw | 50 | 0 | 0.01292 | 0.01581 | max over n = 50: 0.01685 | 0.01685 | 0 | w32 s0 |
| bookcase_side_right | top_z | 2 | 48 | 0.0003164 | 0.0003388 | max over n = 2: 0.0003412 | 0 | -0.0003412 | w46 s0 |
| bookcase_back | cx | 17 | 33 | 9.695e-06 | 1.897e-05 | max over n = 17: 2.372e-05 | 0 | -2.372e-05 | w2 s0 |
| bookcase_back | cy | 17 | 33 | 3.087e-07 | 6.69e-07 | max over n = 17: 6.956e-07 | 6.623e-07 | -6.956e-07 | w0 s0 |
| bookcase_back | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | sx | 50 | 0 | 1.705e-05 | 9.531e-05 | max over n = 50: 0.0001405 | 2.018e-06 | -0.0001405 | w35 s0 |
| bookcase_back | sy | 17 | 33 | 3.769e-07 | 4.554e-06 | max over n = 17: 4.852e-06 | 4.852e-06 | -1.3e-06 | w1 s0 |
| bookcase_back | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | yaw | 50 | 0 | 0.002105 | 0.003187 | max over n = 50: 0.003325 | 0.003325 | -0.00327 | w46 s0 |
| bookcase_back | top_z | 0 | 50 | - | - | - | - | - | - |
| pan_table | cx | 50 | 0 | 1.71e-05 | 2.259e-05 | max over n = 50: 2.43e-05 | 2.43e-05 | 0 | w17 s0 |
| pan_table | cy | 50 | 0 | 1.354e-06 | 4.333e-06 | max over n = 50: 6.379e-06 | 4.746e-06 | -6.379e-06 | w9 s0 |
| pan_table | cz | 0 | 50 | - | - | - | - | - | - |
| pan_table | sx | 50 | 0 | 3.332e-05 | 4.444e-05 | max over n = 50: 4.974e-05 | 0 | -4.974e-05 | w17 s0 |
| pan_table | sy | 50 | 0 | 2.287e-05 | 2.891e-05 | max over n = 50: 3.137e-05 | 0 | -3.137e-05 | w44 s0 |
| pan_table | sz | 0 | 50 | - | - | - | - | - | - |
| pan_table | yaw | 50 | 0 | 0.000208 | 0.0006693 | max over n = 50: 0.0008469 | 0.0008469 | -0.000773 | w9 s0 |
| pan_table | top_z | 50 | 0 | 2.183e-05 | 3.333e-05 | max over n = 50: 3.635e-05 | 0 | -3.635e-05 | w40 s0 |

## box_bookcase / zed2i_lab

| entity | field | n obs | n unobs | p50 | p95 | max over n | max + | max - | worst world |
|---|---|---|---|---|---|---|---|---|---|
| crate | cx | 200 | 0 | 0.000791 | 0.003302 | max over n = 200: 0.007191 | 0.002397 | -0.007191 | w19 s0 |
| crate | cy | 200 | 0 | 0.0009263 | 0.003864 | max over n = 200: 0.005064 | 0.003044 | -0.005064 | w43 s2 |
| crate | cz | 200 | 0 | 0.0008889 | 0.001937 | max over n = 200: 0.002357 | 0 | -0.002357 | w18 s3 |
| crate | sx | 200 | 0 | 0.003268 | 0.006785 | max over n = 200: 0.01504 | 0.007424 | -0.01504 | w18 s1 |
| crate | sy | 200 | 0 | 0.002977 | 0.008104 | max over n = 200: 0.01026 | 0.01026 | -0.00748 | w35 s0 |
| crate | sz | 200 | 0 | 0.003043 | 0.005163 | max over n = 200: 0.005825 | 0 | -0.005825 | w1 s3 |
| crate | yaw | 200 | 0 | 0.3657 | 1.062 | max over n = 200: 1.527 | 1.527 | -1.28 | w13 s3 |
| crate | top_z | 200 | 0 | 0.002422 | 0.004448 | max over n = 200: 0.005263 | 0 | -0.005263 | w18 s3 |
| counter | cx | 116 | 84 | 0.000656 | 0.002337 | max over n = 116: 0.007416 | 0.002001 | -0.007416 | w1 s1 |
| counter | cy | 93 | 107 | 0.001065 | 0.004441 | max over n = 93: 0.007964 | 0.007964 | -0.004617 | w32 s3 |
| counter | cz | 0 | 200 | - | - | - | - | - | - |
| counter | sx | 173 | 27 | 0.002689 | 0.009998 | max over n = 173: 0.01943 | 0.004568 | -0.01943 | w1 s0 |
| counter | sy | 93 | 107 | 0.004615 | 0.007986 | max over n = 93: 0.01869 | 0.00837 | -0.01869 | w15 s1 |
| counter | sz | 0 | 200 | - | - | - | - | - | - |
| counter | yaw | 200 | 0 | 0.2367 | 0.844 | max over n = 200: 1.663 | 1.663 | -1.508 | w22 s3 |
| counter | top_z | 200 | 0 | 0.003642 | 0.004968 | max over n = 200: 0.005807 | 0.003318 | -0.005807 | w20 s3 |
| bookcase_l1_floor | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | yaw | 1 | 199 | 0.1948 | 0.1948 | max over n = 1: 0.1948 | 0 | -0.1948 | w24 s3 |
| bookcase_l1_floor | top_z | 200 | 0 | 0.0008361 | 0.001571 | max over n = 200: 0.002645 | 0.001325 | -0.002645 | w27 s3 |
| bookcase_l1_base | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | yaw | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | top_z | 200 | 0 | 0.00166 | 0.00305 | max over n = 200: 0.003149 | 0.001957 | -0.003149 | w13 s2 |
| bookcase_plinth | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | sy | 200 | 0 | 0.005854 | 0.007333 | max over n = 200: 0.007892 | 0.007892 | -0.006967 | w35 s2 |
| bookcase_plinth | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | yaw | 200 | 0 | 0.1788 | 0.5279 | max over n = 200: 0.5742 | 0.5742 | -0.479 | w38 s1 |
| bookcase_plinth | top_z | 6 | 194 | 0.01108 | 0.01442 | max over n = 6: 0.01482 | 0 | -0.01482 | w46 s2 |
| bookcase_mid | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | yaw | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | top_z | 200 | 0 | 0.001174 | 0.002102 | max over n = 200: 0.002287 | 0.002287 | -0.001589 | w4 s1 |
| bookcase_top | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | yaw | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | top_z | 200 | 0 | 0.0005632 | 0.0009269 | max over n = 200: 0.0009509 | 0.0009509 | -0.0007364 | w47 s3 |
| bookcase_side_left | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | cy | 1 | 199 | 0.0005615 | 0.0005615 | max over n = 1: 0.0005615 | 0 | -0.0005615 | w41 s1 |
| bookcase_side_left | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | sy | 200 | 0 | 0.003553 | 0.0105 | max over n = 200: 0.01079 | 0.006254 | -0.01079 | w35 s1 |
| bookcase_side_left | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | yaw | 200 | 0 | 0.1501 | 0.3867 | max over n = 200: 0.45 | 0.1113 | -0.45 | w4 s2 |
| bookcase_side_left | top_z | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | sy | 200 | 0 | 0.003866 | 0.007781 | max over n = 200: 0.008086 | 0.00516 | -0.008086 | w17 s1 |
| bookcase_side_right | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | yaw | 200 | 0 | 0.159 | 0.2958 | max over n = 200: 0.3303 | 0.3303 | -0.08066 | w7 s1 |
| bookcase_side_right | top_z | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | cx | 1 | 199 | 0.001226 | 0.001226 | max over n = 1: 0.001226 | 0 | -0.001226 | w41 s0 |
| bookcase_back | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | sx | 200 | 0 | 0.002187 | 0.004093 | max over n = 200: 0.00537 | 0.003493 | -0.00537 | w7 s2 |
| bookcase_back | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | yaw | 200 | 0 | 0.05537 | 0.1084 | max over n = 200: 0.1177 | 0.1177 | -0.09929 | w35 s3 |
| bookcase_back | top_z | 0 | 200 | - | - | - | - | - | - |
| pan_table | cx | 0 | 200 | - | - | - | - | - | - |
| pan_table | cy | 0 | 200 | - | - | - | - | - | - |
| pan_table | cz | 0 | 200 | - | - | - | - | - | - |
| pan_table | sx | 0 | 200 | - | - | - | - | - | - |
| pan_table | sy | 0 | 200 | - | - | - | - | - | - |
| pan_table | sz | 0 | 200 | - | - | - | - | - | - |
| pan_table | yaw | 40 | 160 | 0.1772 | 0.5271 | max over n = 40: 0.5994 | 0.5994 | -0.5665 | w23 s2 |
| pan_table | top_z | 200 | 0 | 0.0009638 | 0.00166 | max over n = 200: 0.001743 | 0.001064 | -0.001743 | w16 s2 |

Half-widths for err_table_from_study: `halfwidth_<noise>.json` = max |error| over n per (entity, field), yaw in radians, n = observed count.
