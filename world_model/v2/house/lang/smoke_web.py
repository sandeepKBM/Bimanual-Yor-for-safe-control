"""scripted aiohttp client for the web teleop language panel (no browser).  Server: ... web.server --lang live|mock --port 8767 --token T
    python -m world_model.v2.house.lang.smoke_web --port 8767 --token T --instruction "..." --out DIR
Checks: hello carries `lang`; the sim keeps ticking while the VLM thread runs; the final `lang` message (status, grounded names, spec, mu, latency); the telemetry after."""
import argparse, asyncio, base64, json, time
import aiohttp


async def main(a):
    t0 = time.time()
    tel, lang_msgs, hello, n_tel_during = {}, [], None, 0
    async with aiohttp.ClientSession() as ses:
        async with ses.ws_connect(f"http://127.0.0.1:{a.port}/ws?token={a.token}", max_msg_size=1 << 25, headers={"Origin": f"http://localhost:{a.port}"}) as ws:
            running = {"v": False}

            async def rx():
                nonlocal tel, hello, n_tel_during
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(m.data)
                        if d["type"] == "hello":
                            hello = d
                        elif d["type"] == "tel":
                            tel = d
                            if running["v"]:
                                n_tel_during += 1
                        elif d["type"] == "lang":
                            lang_msgs.append((time.time(), d))
            r = asyncio.ensure_future(rx())
            while not tel.get("ready"):
                await asyncio.sleep(0.3)
                if time.time() - t0 > 300:
                    raise SystemExit("sim never ready")
            print("ready after", round(time.time() - t0, 1), "s; hello.lang =", hello.get("lang"), flush=True)
            for k in range(3):
                await ws.send_str(json.dumps(dict(type="input", src="kbd", seq=k + 1, keys=[], focus=True))); await asyncio.sleep(0.05)
            pose0, sim_t0 = tel["pose"][:], tel["t"]
            ts = time.time(); running["v"] = True
            await ws.send_str(json.dumps(dict(type="lang", instruction=a.instruction)))
            final = None
            while time.time() - ts < a.timeout:
                await asyncio.sleep(0.5)
                if lang_msgs and lang_msgs[-1][1].get("status") in ("done", "error"):
                    final = lang_msgs[-1][1]; break
            running["v"] = False
            print("statuses:", [m[1].get("status") for m in lang_msgs], "wall", round(time.time() - ts, 1), "s", flush=True)
            print("telemetry messages during the VLM call:", n_tel_during, "sim time advanced", round(tel["t"] - sim_t0, 2), "s; pose moved", [round(x - y, 4) for x, y in zip(tel["pose"], pose0)], flush=True)
            if final:
                jpg = final.pop("marks_jpg", None)
                if jpg:
                    open(f"{a.out}/web_marks.jpg", "wb").write(base64.b64decode(jpg))
                json.dump(final, open(f"{a.out}/web_lang.json", "w"), indent=1)
                h = final.get("hud") or {}
                print(json.dumps({k: h.get(k) for k in ("grounded", "spec_status", "checks", "clearances", "unresolved", "mu", "latency", "total_wall_s", "source")}, indent=1)[:2500])
                print(h.get("spec_pretty"))
                print("error:", final.get("error"))
            await asyncio.sleep(1.0)
            print("after: shield", tel.get("shield"), "label", tel.get("shield_label"), flush=True)
            r.cancel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767); ap.add_argument("--token", required=True)
    ap.add_argument("--instruction", default="bring the coffee to my desk without spilling and keep it 10 cm away from the lamp")
    ap.add_argument("--timeout", type=float, default=300.0); ap.add_argument("--out", default=".")
    asyncio.run(main(ap.parse_args()))
