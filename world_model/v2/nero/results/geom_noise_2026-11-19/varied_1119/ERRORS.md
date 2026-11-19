# Depth box fit: error table (sampled)

Guarantee: **sampled** over the drawn worlds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] x seeds [0, 1, 2, 3] (250 runs).  Cameras: ['room_front', 'room_left', 'room_back', 'room_right']; image size [2208, 1242] (w, h).
Truth = geom poses read from the model.  Errors in metres, yaw in degrees.  "max over n" = max |error| over the n observed fits (a measured maximum, not a bound).

Noise model (zed2i_lab): {"sigma_z": "z^2*SIGMA_D_PX/(f_px*BASELINE_M)", "SIGMA_D_PX": 0.2, "BASELINE_M": 0.12, "LAB_CORR_FRAC": 0.5, "LAB_CORR_PX_720": 15, "LAB_DROP": 0.05, "LAB_CALIB_M": 0.003, "LAB_CALIB_DEG": 0.1, "status": "ASSUMED (user approved 11-19) except BASELINE_M = datasheet"}
Clean: no noise, deterministic, seed 0 only.

## box_bookcase / clean

| entity | field | n obs | n unobs | p50 | p95 | max over n | max + | max - | worst world |
|---|---|---|---|---|---|---|---|---|---|
| crate | cx | 49 | 1 | 0.000122 | 0.00017 | max over n = 49: 0.0001733 | 0 | -0.0001733 | w43 s0 |
| crate | cy | 49 | 1 | 1.892e-05 | 0.0001027 | max over n = 49: 0.0004008 | 0.0004008 | -8.466e-05 | w30 s0 |
| crate | cz | 50 | 0 | 0.000614 | 0.0008333 | max over n = 50: 0.0008761 | 0.0008761 | 0 | w42 s0 |
| crate | sx | 49 | 1 | 0.0004647 | 0.0005201 | max over n = 49: 0.0005366 | 0 | -0.0005366 | w46 s0 |
| crate | sy | 50 | 0 | 0.001175 | 0.001414 | max over n = 50: 0.001915 | 0 | -0.001915 | w30 s0 |
| crate | sz | 50 | 0 | 0.0001316 | 0.0004755 | max over n = 50: 0.0005598 | 0.0003445 | -0.0005598 | w42 s0 |
| crate | yaw | 50 | 0 | 0.01434 | 0.02495 | max over n = 50: 0.03197 | 0.02501 | -0.03197 | w5 s0 |
| crate | top_z | 50 | 0 | 0.0005972 | 0.0006117 | max over n = 50: 0.0006163 | 0.0006163 | 0 | w11 s0 |
| counter | cx | 50 | 0 | 0.0001221 | 0.001666 | max over n = 50: 0.00193 | 0.00193 | -0.0001848 | w16 s0 |
| counter | cy | 39 | 11 | 5.338e-05 | 0.0001353 | max over n = 39: 0.0001425 | 0.0001096 | -0.0001425 | w28 s0 |
| counter | cz | 0 | 50 | - | - | - | - | - | - |
| counter | sx | 50 | 0 | 0.0007835 | 0.00404 | max over n = 50: 0.004576 | 0 | -0.004576 | w16 s0 |
| counter | sy | 39 | 11 | 0.0009603 | 0.001096 | max over n = 39: 0.001118 | 0 | -0.001118 | w16 s0 |
| counter | sz | 0 | 50 | - | - | - | - | - | - |
| counter | yaw | 50 | 0 | 0.00866 | 0.01789 | max over n = 50: 0.02142 | 0.0154 | -0.02142 | w23 s0 |
| counter | top_z | 50 | 0 | 0.0004669 | 0.0005572 | max over n = 50: 0.0006531 | 0.0006531 | -0.0004268 | w39 s0 |
| bookcase_l1_floor | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | cy | 9 | 41 | 0.02757 | 0.0419 | max over n = 9: 0.04387 | 0.03642 | -0.04387 | w17 s0 |
| bookcase_l1_floor | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | sy | 50 | 0 | 0.001178 | 0.001248 | max over n = 50: 0.001263 | 0 | -0.001263 | w9 s0 |
| bookcase_l1_floor | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_floor | yaw | 50 | 0 | 0.002885 | 0.008294 | max over n = 50: 0.01469 | 0.01469 | -0.01061 | w39 s0 |
| bookcase_l1_floor | top_z | 50 | 0 | 0.0005779 | 0.0005856 | max over n = 50: 0.000587 | 0.000587 | 0 | w10 s0 |
| bookcase_l1_base | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | sy | 1 | 49 | 1.532 | 1.532 | max over n = 1: 1.532 | 0 | -1.532 | w2 s0 |
| bookcase_l1_base | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_l1_base | yaw | 50 | 0 | 0.006362 | 0.1989 | max over n = 50: 0.2147 | 0.1953 | -0.2147 | w35 s0 |
| bookcase_l1_base | top_z | 50 | 0 | 0.0006435 | 0.000654 | max over n = 50: 0.0006555 | 0.0006555 | 0 | w27 s0 |
| bookcase_plinth | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | cy | 10 | 40 | 0.02821 | 0.04493 | max over n = 10: 0.04586 | 0.04586 | -0.04379 | w42 s0 |
| bookcase_plinth | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | sy | 50 | 0 | 0.0012 | 0.001248 | max over n = 50: 0.001255 | 0 | -0.001255 | w22 s0 |
| bookcase_plinth | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_plinth | yaw | 50 | 0 | 0.002469 | 0.006111 | max over n = 50: 0.006238 | 0.002478 | -0.006238 | w7 s0 |
| bookcase_plinth | top_z | 1 | 49 | 0.0171 | 0.0171 | max over n = 1: 0.0171 | 0 | -0.0171 | w41 s0 |
| bookcase_mid | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | sy | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_mid | yaw | 50 | 0 | 0.005862 | 0.006639 | max over n = 50: 0.04786 | 0.01571 | -0.04786 | w37 s0 |
| bookcase_mid | top_z | 50 | 0 | 0.0005508 | 0.0005613 | max over n = 50: 0.0005642 | 0.0005642 | 0 | w27 s0 |
| bookcase_top | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | cy | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | sy | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_top | yaw | 50 | 0 | 0.006169 | 0.02889 | max over n = 50: 0.04621 | 0.04621 | -0.03369 | w8 s0 |
| bookcase_top | top_z | 50 | 0 | 0.0004881 | 0.0004973 | max over n = 50: 0.0005004 | 0.0005004 | 0 | w32 s0 |
| bookcase_side_left | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | cy | 10 | 40 | 0.02853 | 0.04485 | max over n = 10: 0.04547 | 0.04547 | -0.04409 | w42 s0 |
| bookcase_side_left | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | sy | 50 | 0 | 0.001222 | 0.001263 | max over n = 50: 0.001272 | 0 | -0.001272 | w22 s0 |
| bookcase_side_left | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_left | yaw | 50 | 0 | 0.01384 | 0.0151 | max over n = 50: 0.01614 | 0 | -0.01614 | w5 s0 |
| bookcase_side_left | top_z | 50 | 0 | 0.0008536 | 0.003449 | max over n = 50: 0.003488 | 0 | -0.003488 | w14 s0 |
| bookcase_side_right | cx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | cy | 9 | 41 | 0.02716 | 0.04132 | max over n = 9: 0.04324 | 0.03692 | -0.04324 | w17 s0 |
| bookcase_side_right | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | sx | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | sy | 50 | 0 | 0.001247 | 0.001322 | max over n = 50: 0.001327 | 0 | -0.001327 | w9 s0 |
| bookcase_side_right | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_side_right | yaw | 50 | 0 | 0.01333 | 0.0145 | max over n = 50: 0.01496 | 0.01496 | 0 | w41 s0 |
| bookcase_side_right | top_z | 50 | 0 | 0.0008081 | 0.00343 | max over n = 50: 0.00354 | 0 | -0.00354 | w24 s0 |
| bookcase_back | cx | 18 | 32 | 3.509e-05 | 0.003353 | max over n = 18: 0.004906 | 0.003078 | -0.004906 | w42 s0 |
| bookcase_back | cy | 14 | 36 | 2.513e-06 | 7.166e-06 | max over n = 14: 7.373e-06 | 3.735e-06 | -7.373e-06 | w0 s0 |
| bookcase_back | cz | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | sx | 50 | 0 | 0.0005881 | 0.0007572 | max over n = 50: 0.00167 | 0 | -0.00167 | w15 s0 |
| bookcase_back | sy | 14 | 36 | 0.000889 | 0.0009127 | max over n = 14: 0.0009194 | 0 | -0.0009194 | w39 s0 |
| bookcase_back | sz | 0 | 50 | - | - | - | - | - | - |
| bookcase_back | yaw | 50 | 0 | 0.001903 | 0.00373 | max over n = 50: 0.004613 | 0.003774 | -0.004613 | w36 s0 |
| bookcase_back | top_z | 40 | 10 | 0.0006036 | 0.003463 | max over n = 40: 0.003518 | 0 | -0.003518 | w32 s0 |
| pan_table | cx | 48 | 2 | 7.243e-05 | 0.0001109 | max over n = 48: 0.0002605 | 0.0002605 | -0.0001642 | w45 s0 |
| pan_table | cy | 50 | 0 | 1.108e-05 | 2.68e-05 | max over n = 50: 3.521e-05 | 2.541e-05 | -3.521e-05 | w17 s0 |
| pan_table | cz | 0 | 50 | - | - | - | - | - | - |
| pan_table | sx | 48 | 2 | 0.0004779 | 0.0007946 | max over n = 48: 0.001119 | 0 | -0.001119 | w45 s0 |
| pan_table | sy | 50 | 0 | 0.001055 | 0.001072 | max over n = 50: 0.001084 | 0 | -0.001084 | w17 s0 |
| pan_table | sz | 0 | 50 | - | - | - | - | - | - |
| pan_table | yaw | 50 | 0 | 0.0007683 | 0.001872 | max over n = 50: 0.002 | 0.0007762 | -0.002 | w14 s0 |
| pan_table | top_z | 50 | 0 | 0.0006196 | 0.0006289 | max over n = 50: 0.0006368 | 0.0006368 | 0 | w16 s0 |

## box_bookcase / zed2i_lab

| entity | field | n obs | n unobs | p50 | p95 | max over n | max + | max - | worst world |
|---|---|---|---|---|---|---|---|---|---|
| crate | cx | 200 | 0 | 0.0007568 | 0.00312 | max over n = 200: 0.007178 | 0.002421 | -0.007178 | w19 s0 |
| crate | cy | 200 | 0 | 0.0009093 | 0.00381 | max over n = 200: 0.005071 | 0.005071 | -0.004898 | w45 s3 |
| crate | cz | 200 | 0 | 0.0006376 | 0.001659 | max over n = 200: 0.00349 | 0.0003646 | -0.00349 | w37 s1 |
| crate | sx | 200 | 0 | 0.003199 | 0.007192 | max over n = 200: 0.01538 | 0.007032 | -0.01538 | w18 s1 |
| crate | sy | 200 | 0 | 0.002529 | 0.007409 | max over n = 200: 0.01205 | 0.008905 | -0.01205 | w9 s1 |
| crate | sz | 200 | 0 | 0.002556 | 0.004911 | max over n = 200: 0.009008 | 0 | -0.009008 | w37 s1 |
| crate | yaw | 200 | 0 | 0.3767 | 1.069 | max over n = 200: 1.518 | 1.518 | -1.265 | w13 s3 |
| crate | top_z | 200 | 0 | 0.001916 | 0.004114 | max over n = 200: 0.007994 | 1.643e-05 | -0.007994 | w37 s1 |
| counter | cx | 170 | 30 | 0.001002 | 0.006412 | max over n = 170: 0.01091 | 0.003176 | -0.01091 | w41 s3 |
| counter | cy | 96 | 104 | 0.0009552 | 0.004278 | max over n = 96: 0.006924 | 0.006088 | -0.006924 | w1 s1 |
| counter | cz | 0 | 200 | - | - | - | - | - | - |
| counter | sx | 172 | 28 | 0.002883 | 0.0102 | max over n = 172: 0.01934 | 0.005089 | -0.01934 | w1 s0 |
| counter | sy | 96 | 104 | 0.00418 | 0.008607 | max over n = 96: 0.02049 | 0.007303 | -0.02049 | w15 s1 |
| counter | sz | 0 | 200 | - | - | - | - | - | - |
| counter | yaw | 200 | 0 | 0.2454 | 0.9012 | max over n = 200: 1.708 | 1.708 | -1.513 | w22 s3 |
| counter | top_z | 200 | 0 | 0.003553 | 0.006567 | max over n = 200: 0.008564 | 0.002542 | -0.008564 | w47 s2 |
| bookcase_l1_floor | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_floor | yaw | 12 | 188 | 0.6634 | 1.919 | max over n = 12: 2.153 | 0.583 | -2.153 | w45 s3 |
| bookcase_l1_floor | top_z | 200 | 0 | 0.0005062 | 0.001606 | max over n = 200: 0.002001 | 0.001972 | -0.002001 | w27 s3 |
| bookcase_l1_base | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | yaw | 0 | 200 | - | - | - | - | - | - |
| bookcase_l1_base | top_z | 200 | 0 | 0.001877 | 0.002459 | max over n = 200: 0.002652 | 0.002652 | -0.002457 | w35 s1 |
| bookcase_plinth | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | cy | 41 | 159 | 0.02626 | 0.0426 | max over n = 41: 0.04886 | 0.04886 | -0.0426 | w42 s0 |
| bookcase_plinth | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | sy | 200 | 0 | 0.004964 | 0.007724 | max over n = 200: 0.008052 | 0.00688 | -0.008052 | w35 s1 |
| bookcase_plinth | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_plinth | yaw | 200 | 0 | 0.1833 | 0.5431 | max over n = 200: 0.5774 | 0.5774 | -0.4853 | w23 s1 |
| bookcase_plinth | top_z | 200 | 0 | 0.01054 | 0.01887 | max over n = 200: 0.02582 | 0 | -0.02582 | w14 s0 |
| bookcase_mid | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_mid | yaw | 6 | 194 | 0.141 | 0.1793 | max over n = 6: 0.1839 | 0.1839 | -0.1655 | w1 s1 |
| bookcase_mid | top_z | 200 | 0 | 0.0006279 | 0.002671 | max over n = 200: 0.002857 | 0.002857 | -0.001017 | w4 s1 |
| bookcase_top | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | cy | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | sy | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | yaw | 0 | 200 | - | - | - | - | - | - |
| bookcase_top | top_z | 200 | 0 | 0.001015 | 0.001417 | max over n = 200: 0.001438 | 0.001438 | -0.0002369 | w1 s3 |
| bookcase_side_left | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | cy | 42 | 158 | 0.0321 | 0.04228 | max over n = 42: 0.04476 | 0.04231 | -0.04476 | w1 s1 |
| bookcase_side_left | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | sy | 200 | 0 | 0.002614 | 0.0114 | max over n = 200: 0.01167 | 0.005293 | -0.01167 | w4 s1 |
| bookcase_side_left | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_left | yaw | 200 | 0 | 0.1452 | 0.3797 | max over n = 200: 0.4407 | 0.1169 | -0.4407 | w4 s2 |
| bookcase_side_left | top_z | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | cx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | cy | 35 | 165 | 0.02475 | 0.04214 | max over n = 35: 0.04557 | 0.04557 | -0.04496 | w42 s0 |
| bookcase_side_right | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | sx | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | sy | 200 | 0 | 0.003001 | 0.008673 | max over n = 200: 0.008968 | 0.004256 | -0.008968 | w17 s1 |
| bookcase_side_right | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_side_right | yaw | 200 | 0 | 0.1405 | 0.275 | max over n = 200: 0.3119 | 0.3119 | -0.09822 | w7 s1 |
| bookcase_side_right | top_z | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | cx | 52 | 148 | 0.002683 | 0.02068 | max over n = 52: 0.02827 | 0.02827 | -0.005961 | w21 s1 |
| bookcase_back | cy | 18 | 182 | 0.1585 | 0.313 | max over n = 18: 0.3556 | 0.1798 | -0.3556 | w21 s1 |
| bookcase_back | cz | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | sx | 200 | 0 | 0.002601 | 0.004481 | max over n = 200: 0.006008 | 0.003027 | -0.006008 | w7 s2 |
| bookcase_back | sy | 18 | 182 | 1.253 | 1.326 | max over n = 18: 1.373 | 0 | -1.373 | w5 s3 |
| bookcase_back | sz | 0 | 200 | - | - | - | - | - | - |
| bookcase_back | yaw | 200 | 0 | 0.05616 | 0.1077 | max over n = 200: 0.1183 | 0.1183 | -0.1012 | w35 s3 |
| bookcase_back | top_z | 0 | 200 | - | - | - | - | - | - |
| pan_table | cx | 0 | 200 | - | - | - | - | - | - |
| pan_table | cy | 14 | 186 | 0.001545 | 0.002981 | max over n = 14: 0.003278 | 0.001182 | -0.003278 | w45 s3 |
| pan_table | cz | 0 | 200 | - | - | - | - | - | - |
| pan_table | sx | 0 | 200 | - | - | - | - | - | - |
| pan_table | sy | 14 | 186 | 0.005482 | 0.006751 | max over n = 14: 0.008043 | 0 | -0.008043 | w16 s1 |
| pan_table | sz | 0 | 200 | - | - | - | - | - | - |
| pan_table | yaw | 114 | 86 | 0.1279 | 0.4067 | max over n = 114: 1.703 | 1.703 | -0.5622 | w24 s2 |
| pan_table | top_z | 200 | 0 | 0.0005327 | 0.001543 | max over n = 200: 0.001675 | 0.001675 | -0.001118 | w9 s1 |

Half-widths for err_table_from_study: `halfwidth_<noise>.json` = max |error| over n per (entity, field), yaw in radians, n = observed count.
