import sys
import numpy as np
import imageio.v2 as imageio

path = sys.argv[1]
d = np.load(path)
for k in d.files:
    print(k, d[k].shape, d[k].dtype)

if len(sys.argv) > 2:
    out_png = sys.argv[2]
    t_mid = d["rgb_head"].shape[0] // 2
    frame_a = d["rgb_head"][t_mid]
    frame_b = d["rgb_wrist_right"][t_mid]
    frame_first_a = d["rgb_head"][0]
    montage = np.concatenate([frame_first_a, frame_a, frame_b], axis=1)
    imageio.imwrite(out_png, montage)
    print(f"wrote {out_png} (t=0 cup_cam_a | t=mid cup_cam_a | t=mid cup_cam_b)")
    print("spilled per-step (dropped key) any-true count:", int(d["dropped"].sum()))
    print("collision any-true count:", int(d["collision"].sum()))
