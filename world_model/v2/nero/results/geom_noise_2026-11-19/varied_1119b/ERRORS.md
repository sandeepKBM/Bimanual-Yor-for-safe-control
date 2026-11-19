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
| counter | cx | 50 | 0 | 1.321e-05 | 3.319e-05 | max over n = 50: 8.681e-05 | 4.123e-05 | -8.681e-05 | w29 s0 |
| counter | cy | 36 | 14 | 1.045e-05 | 2.888e-05 | max over n = 36: 3.121e-05 | 2.858e-05 | -3.121e-05 | w19 s0 |
| counter | cz | 0 | 50 | - | - | - | - | - | - |
| counter | sx | 50 | 0 | 1.638e-05 | 4.224e-05 | max over n = 50: 0.0002332 | 4.773e-05 | -0.0002332 | w29 s0 |
| counter | sy | 36 | 14 | 4.769e-05 | 8.877e-05 | max over n = 36: 0.0001549 | 0 | -0.0001549 | w43 s0 |
| counter | sz | 0 | 50 | - | - | - | - | - | - |
| counter | yaw | 50 | 0 | 0.003989 | 0.008959 | max over n = 50: 0.0179 | 0.006884 | -0.0179 | w29 s0 |
| counter | top_z | 50 | 0 | 0.000102 | 0.0001501 | max over n = 50: 0.0001528 | 0 | -0.0001528 | w26 s0 |
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
| bookcase_plinth | sy | 50 | 0 | 3.09e-05 | 3.5e-05 | max over n = 50: 3.688e-05 | 0 | -3.688e-05 | w27 s0 |
| bookcase_plinth | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | yaw | 50 | 0 | 0.0002197 | 0.001128 | max over n = 50: 0.001584 | 0.001436 | -0.001584 | w33 s0 |
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
| bookcase_side_left | sy | 50 | 0 | 0.0005935 | 0.0007524 | max over n = 50: 0.0007655 | 0 | -0.0007655 | w4 s0 |
| bookcase_side_left | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | yaw | 50 | 0 | 0.01335 | 0.02096 | max over n = 50: 0.02354 | 0 | -0.02354 | w10 s0 |
| bookcase_side_left | top_z | 1 | 49 | 0.0007623 | 0.0007623 | max over n = 1: 0.0007623 | 0 | -0.0007623 | w5 s0 |
| bookcase_side_right | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | sy | 50 | 0 | 0.0005638 | 0.0007243 | max over n = 50: 0.000773 | 0 | -0.000773 | w46 s0 |
| bookcase_side_right | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | yaw | 50 | 0 | 0.01375 | 0.01983 | max over n = 50: 0.02268 | 0.02268 | 0 | w46 s0 |
| bookcase_side_right | top_z | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | cx | 17 | 33 | 7.921e-06 | 0.0004426 | max over n = 17: 0.0004823 | 1.766e-06 | -0.0004823 | w34 s0 |
| bookcase_back | cy | 17 | 33 | 3.398e-07 | 1.962e-05 | max over n = 17: 2.025e-05 | 6.194e-07 | -2.025e-05 | w34 s0 |
| bookcase_back | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | sx | 50 | 0 | 3.025e-05 | 0.001248 | max over n = 50: 0.001525 | 0 | -0.001525 | w27 s0 |
| bookcase_back | sy | 17 | 33 | 3.778e-07 | 4.553e-06 | max over n = 17: 4.852e-06 | 4.852e-06 | -1.3e-06 | w1 s0 |
| bookcase_back | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | yaw | 50 | 0 | 0.002745 | 0.00443 | max over n = 50: 0.004773 | 0.004773 | -0.004626 | w32 s0 |
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
| crate | cx | 200 | 0 | 0.0007845 | 0.003162 | max over n = 200: 0.007191 | 0.002397 | -0.007191 | w19 s0 |
| crate | cy | 200 | 0 | 0.0009906 | 0.004033 | max over n = 200: 0.006587 | 0.006587 | -0.005252 | w45 s1 |
| crate | cz | 200 | 0 | 0.0008889 | 0.001937 | max over n = 200: 0.002357 | 0 | -0.002357 | w18 s3 |
| crate | sx | 200 | 0 | 0.003268 | 0.006785 | max over n = 200: 0.01504 | 0.007424 | -0.01504 | w18 s1 |
| crate | sy | 200 | 0 | 0.002959 | 0.008104 | max over n = 200: 0.01463 | 0.01003 | -0.01463 | w45 s1 |
| crate | sz | 200 | 0 | 0.003049 | 0.005163 | max over n = 200: 0.005825 | 0 | -0.005825 | w1 s3 |
| crate | yaw | 200 | 0 | 0.3657 | 1.062 | max over n = 200: 1.527 | 1.527 | -1.28 | w13 s3 |
| crate | top_z | 200 | 0 | 0.002422 | 0.004448 | max over n = 200: 0.005263 | 0 | -0.005263 | w18 s3 |
| counter | cx | 162 | 38 | 0.00493 | 0.01422 | max over n = 162: 0.01884 | 0.002425 | -0.01884 | w27 s3 |
| counter | cy | 155 | 45 | 0.003559 | 0.1372 | max over n = 155: 0.1439 | 0.1439 | -0.141 | w10 s2 |
| counter | cz | 0 | 200 | - | - | - | - | - | - |
| counter | sx | 200 | 0 | 0.01099 | 0.03612 | max over n = 200: 0.0449 | 0.00417 | -0.0449 | w7 s1 |
| counter | sy | 155 | 45 | 0.008573 | 0.3921 | max over n = 155: 0.5679 | 0.009556 | -0.5679 | w13 s3 |
| counter | sz | 0 | 200 | - | - | - | - | - | - |
| counter | yaw | 200 | 0 | 0.4571 | 1.686 | max over n = 200: 2.189 | 2.189 | -2.052 | w27 s3 |
| counter | top_z | 196 | 4 | 0.003261 | 0.004969 | max over n = 196: 0.007958 | 0.003318 | -0.007958 | w20 s1 |
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
| bookcase_plinth | sy | 200 | 0 | 0.00627 | 0.007752 | max over n = 200: 0.008322 | 0.008322 | -0.007172 | w35 s2 |
| bookcase_plinth | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | yaw | 200 | 0 | 0.1622 | 0.4715 | max over n = 200: 0.5573 | 0.505 | -0.5573 | w36 s2 |
| bookcase_plinth | top_z | 0 | 200 | - | - | - | - | - | - |
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
| bookcase_side_right | sy | 200 | 0 | 0.003866 | 0.007781 | max over n = 200: 0.008086 | 0.005153 | -0.008086 | w17 s1 |
| bookcase_side_right | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | yaw | 200 | 0 | 0.159 | 0.2958 | max over n = 200: 0.3303 | 0.3303 | -0.08066 | w7 s1 |
| bookcase_side_right | top_z | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | cx | 23 | 177 | 0.003581 | 0.02634 | max over n = 23: 0.02716 | 0.02716 | -0.00236 | w21 s1 |
| bookcase_back | cy | 22 | 178 | 0.1503 | 0.3255 | max over n = 22: 0.3428 | 0.2031 | -0.3428 | w21 s1 |
| bookcase_back | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | sx | 200 | 0 | 0.002187 | 0.004093 | max over n = 200: 0.00537 | 0.003493 | -0.00537 | w7 s2 |
| bookcase_back | sy | 22 | 178 | 1.262 | 1.356 | max over n = 22: 1.392 | 0 | -1.392 | w47 s3 |
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
