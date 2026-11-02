# Drake feasibility spike 2026-10-30 (pydrake 1.58.0, CPU, nice, 1 thread; host load ~22 on 4 cores so wall times are pessimistic)
Code: world_model/v2/nero/pan/drake_spike/{common,unit_tests,scene,passive,robot_api,stl2obj}.py. Raw: *.json here.
Contact: hydroelastic_with_fallback + discrete SAP (kSap). ASSUMED: modulus 5e6 Pa, dissipation 1 s/m, mesh res hint 0.01 m (unit tests) / 0.02 m (pan scene), friction as in MJCF.
Unit tests (analytic twist slip torque 0.153 Nm):
 twist dt=5ms slip 0.1565 Nm, yaw@1s 0.0009 rad ; dt=1ms slip 0.139 Nm, yaw@1s 0.0004 rad
 creep dt=5ms 0.0032 mm/s, slide 0.013 mm ; dt=1ms 0.0012 mm/s, slide 0.005 mm
Passive pan+lid+puck 5 s (robot absent): pan xy drift ~6e-7 mm, z settle -0.11 mm; lid vs pan <0.004 mm; puck sinks 0.92 mm; 2.7 s wall / sim s (dt 5 ms), 13.1 (dt 1 ms)
Full scene (robot loaded, joints locked): 3.3 s/sim s; PD-held arms: 3.0 s/sim s (dt 5 ms)
Exact restore: bitwise identical (3 restore methods, incl. pan kicked at t=1 s).
