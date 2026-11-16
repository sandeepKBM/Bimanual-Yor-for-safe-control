# Peak object tilt of the nominal carry (K4)

Measured: peak object tilt (deg) over hold + lift/carry/place/release, per cell x a_plan, n=20 worlds per row, worlds = carry_cert.sample_world with carry_cert.cell_seed (seeds in TABLE.json). Statistics: p50 / p95 / max over n (a max over n, not a bound).

Noise model: simulation only: nominal carry (no sensing noise, no injected disturbance); worlds i.i.d. uniform on the cell; peak of the object z-axis tilt from vertical.

Leaves out: weld compliance of the real gripper (sim grasp = stiff rigid weld, config.WELD_STIFF), sensing / estimation error, base motion, slosh, controller latency, worlds outside the cell.

planner closed_chain; git HEAD 0debc37d0514f532377d63dde2d3ada4fd0f0b2a; BIMANUAL_* env: {'BIMANUAL_PLANNER': 'closed_chain'}.
Consumer: stack/checks/tilt.py tilt_row (next HIGHER measured level; errored / stopped-early / over-U episodes count as failures; CP bound).
