"""scripted aiohttp WebSocket client for the smoke test (no browser).  python -m world_model.v2.house.teleop.web.smoke_client --out DIR"""
import argparse, asyncio, json, time
import aiohttp


async def main(a):
    t0 = time.time()
    tel = {}; frames = []; seq = 0; sent = {}; lat = []; first = None; log = []
    async with aiohttp.ClientSession() as ses:
        async with ses.ws_connect(f"http://127.0.0.1:{a.port}/ws?token={a.token}", max_msg_size=1 << 24, headers={"Origin": f"http://localhost:{a.port}"}) as ws:
            async def rx():
                nonlocal tel
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.BINARY:
                        frames.append((time.time(), m.data[1:]))
                    elif m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d["type"] == "tel":
                            tel = d
                            ack = d.get("ack", {}).get("kbd", -1)
                            if ack in sent:
                                lat.append((time.time() - sent.pop(ack)) * 1e3)
            r = asyncio.ensure_future(rx())
            async def send(keys, focus=True):
                nonlocal seq
                seq += 1; sent[seq] = time.time()
                await ws.send_str(json.dumps(dict(type="input", src="kbd", seq=seq, keys=keys, focus=focus)))
            async def hold(keys, dur):
                end = time.time() + dur
                while time.time() < end:
                    await send(keys); await asyncio.sleep(0.02)
            while not tel.get("ready"):
                await asyncio.sleep(0.2)
                if time.time() - t0 > 240: raise SystemExit("sim never ready")
            print("ready after", round(time.time() - t0, 1), "s", flush=True)
            await hold([], 1.0)
            if a.assist:                                   # assisted friction grasp via the page's button path (a 120 ms KeyG tap), then place via KeyH
                async def tap(code):
                    await send([code]); await asyncio.sleep(0.12); await send([])
                await send(["Enter"]); await send([]); await asyncio.sleep(0.2)
                print("armed:", tel["enabled"], "assist panel telemetry:", tel.get("assist"), "can_grasp", tel.get("can_grasp"), flush=True)
                await tap("KeyG")
                t1 = time.time(); seen = []
                while time.time() - t1 < a.assist_timeout:
                    await send([]); await asyncio.sleep(0.1)
                    st = (tel.get("assist") or {}).get("state")
                    if not seen or seen[-1] != st: seen.append(st); print(f"  t={tel['t']} wall+{time.time()-t1:.0f}s assist state -> {st}", flush=True)
                    if st in ("HOLD", "FAILED"): break
                print("grasp result:", (tel.get("assist") or {}), "grasped", tel.get("grasped"), "beads_out", tel.get("beads_out"), flush=True)
                if a.place and (tel.get("assist") or {}).get("state") == "HOLD":
                    await hold(["KeyR"], 1.6)                  # lift up ~0.10 m
                    await send([]); await asyncio.sleep(0.5)
                    await tap("KeyH")
                    t1 = time.time(); seen = []
                    while time.time() - t1 < a.assist_timeout:
                        await send([]); await asyncio.sleep(0.1)
                        st = (tel.get("assist") or {}).get("state")
                        if not seen or seen[-1] != st: seen.append(st); print(f"  t={tel['t']} wall+{time.time()-t1:.0f}s place state -> {st}", flush=True)
                        if st == "IDLE" and (tel.get("assist") or {}).get("place") is not None: break
                    print("place result:", (tel.get("assist") or {}), "beads_out", tel.get("beads_out"), flush=True)
                r.cancel()
                if frames:
                    open(f"{a.out}/frame.jpg", "wb").write(frames[-1][1])
                return
            if a.tuck:                                     # tuck / camera / footprint scenario (live footprint server): arm, drive, N (tuck), C (cycle cameras), M (carry)
                async def tap(code, dur=0.12):
                    await send([code]); await asyncio.sleep(dur); await send([])
                print("hello-time telemetry fields:", {k: tel.get(k) for k in ("cam", "tuck", "people", "fp_mode", "fp_w", "fp_l", "fp_ms", "tray_off", "render_ms")}, flush=True)
                await send(["Enter"]); await send([]); await asyncio.sleep(0.2)
                p0 = tel["pose"][:]
                await hold(["KeyW"], 2.0)
                print("drive 2 s: pose delta", [round(x - y, 3) for x, y in zip(tel["pose"], p0)], "fp_w", tel["fp_w"], flush=True)
                await tap("KeyN")
                seen = []; t1 = time.time()
                while time.time() - t1 < a.tuck_timeout:
                    await send([]); await asyncio.sleep(0.1)
                    if not seen or seen[-1] != tel["tuck"]: seen.append(tel["tuck"]); print(f"  tuck state -> {tel['tuck']}  tray_off {tel['tray_off']} fp_w {tel['fp_w']} fp_l {tel['fp_l']} beads_out {tel['beads_out']}", flush=True)
                    if tel["tuck"] == "tucked": break
                await hold([], 1.0)
                print("tucked: fp_w", tel["fp_w"], "beads_out", tel["beads_out"], "cup", tel["cup_state"], "tilt", tel["cup_tilt"], flush=True)
                cams = []
                for i in range(5):
                    tf = len(frames)
                    cams.append(tel["cam"])
                    await asyncio.sleep(1.0)
                    if len(frames) > tf:
                        open(f"{a.out}/frame_{tel['cam']}.jpg", "wb").write(frames[-1][1])
                    await tap("KeyC")
                    await asyncio.sleep(0.3)
                print("cameras cycled:", cams, flush=True)
                await tap("KeyM")
                t1 = time.time(); seen = []
                while time.time() - t1 < a.tuck_timeout:
                    await send([]); await asyncio.sleep(0.1)
                    if not seen or seen[-1] != tel["tuck"]: seen.append(tel["tuck"]); print(f"  carry state -> {tel['tuck']}", flush=True)
                    if tel["tuck"] == "carry": break
                await hold([], 0.5)
                r.cancel()
                ft = [f[0] for f in frames]
                print("frames", len(frames), "fps", round((len(ft) - 1) / (ft[-1] - ft[0]), 1) if len(ft) > 2 else None)
                print("final tel:", {k: tel.get(k) for k in ("rtf", "tick_ms", "render_fps", "render_ms", "beads_out", "cup_state", "fp_mode", "fp_w", "fp_ms", "tuck", "cam", "people", "impl")})
                return
            p0 = tel["pose"][:]; log.append(("disarmed hold W", None))
            await hold(["KeyW"], 1.0)
            print("disarmed + W: pose moved", [round(x - y, 3) for x, y in zip(tel["pose"], p0)], "enabled", tel["enabled"], flush=True)
            await send(["Enter"]); await send([]); await asyncio.sleep(0.1)
            print("armed:", tel["enabled"], flush=True)
            p0 = tel["pose"][:]; ta = tel["t"]; tw = time.time()
            await hold(["KeyW"], 3.0)
            print("drive 3 s wall: pose delta", [round(x - y, 3) for x, y in zip(tel["pose"], p0)], "speed", tel["speed"], "sim dt", round(tel["t"] - ta, 2), "rtf", tel["rtf"], flush=True)
            await send([]); await hold([], 0.8)
            print("after release: speed", tel["speed"], flush=True)
            await hold(["ArrowUp", "KeyT"], 1.5)
            print("tray jog: cmd", tel["cmd_twist"], "tick_ms", tel["tick_ms"], flush=True)
            # watchdog: hold W then go silent
            await hold(["KeyW"], 1.0)
            ps = tel["pose"][:]
            await asyncio.sleep(0.6)    # > 250 ms silent
            print("silent 0.6 s: watchdog", tel["watchdog"], "cmd", tel["cmd_twist"], "safe", tel["safe_twist"], flush=True)
            await asyncio.sleep(1.0)
            print("later: speed", tel["speed"], "pose delta since silence", [round(x - y, 3) for x, y in zip(tel["pose"], ps)], flush=True)
            await hold(["Escape"], 0.1); await hold([], 0.3)
            print("Esc latched:", tel["latched"], "enabled", tel["enabled"], flush=True)
            await send(["Enter"]); await send([]); await asyncio.sleep(0.1)
            print("Enter re-armed:", tel["enabled"], flush=True)
            r.cancel()
    ft = [f[0] for f in frames]
    print("frames", len(frames), "fps", round((len(ft) - 1) / (ft[-1] - ft[0]), 1) if len(ft) > 2 else None)
    print("ack latency ms: median", round(sorted(lat)[len(lat)//2], 1), "p95", round(sorted(lat)[int(len(lat)*.95)], 1), "n", len(lat))
    print("final tel:", {k: tel[k] for k in ("rtf", "tick_ms", "render_fps", "beads_out", "cup_state", "wall_clr", "interv_frac", "impl")})
    open(f"{a.out}/frame.jpg", "wb").write(frames[-1][1])
    import cv2, numpy as np
    img = cv2.imdecode(np.frombuffer(frames[-1][1], np.uint8), 1); cv2.imwrite(f"{a.out}/frame.png", img); print("png", img.shape)

ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--token", default="smoke123"); ap.add_argument("--out", required=True)
ap.add_argument("--assist", action="store_true", help="assisted friction-grasp scenario (server: --task coffee --grasp-mode friction --spawn-at cafe_counter_pick; --task carry has no counter)")
ap.add_argument("--tuck", action="store_true", help="tuck / camera-cycle / live-footprint scenario"); ap.add_argument("--tuck-timeout", type=float, default=60.0)
ap.add_argument("--place", action="store_true"); ap.add_argument("--assist-timeout", type=float, default=240.0)
asyncio.run(main(ap.parse_args()))
