"""vlm/replay_service.py -- replay a runner-format request JSONL through the running file-queue daemon (vlm/service.py) and write
runner-format output, so served models (e.g. Cosmos 3 behind vLLM) get the exact prompts the batch runner gave the HF models.

    BIMANUAL_VLM_SVC=/common/users/ss5772/bimanual_data/vlm/svc_cosmos3-edge \
    python -m world_model.v2.vlm.replay_service --req <DATA>/req/specv2.jsonl --out <DATA>/out/specv2_cosmos3-edge.jsonl
then: python -m world_model.v2.vlm.analyze (tag specv2).  Resumes from <out>.partial; writes <out>.DONE last.
"""
import argparse, json, os, sys, time
from pathlib import Path

from world_model.v2.vlm import service


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    st = service.status()
    if not st.get("up"):
        sys.exit(f"daemon not up at {service.SVC}: {st}")
    out = Path(a.out)
    part = Path(str(out) + ".partial")
    done = {json.loads(l)["id"] for l in open(part) if l.strip()} if part.exists() else set()
    reqs = [json.loads(l) for l in open(a.req) if l.strip()]
    t0 = time.time()
    with open(part, "a") as f:
        for i, r in enumerate(reqs):
            if r["id"] in done:
                continue
            o = service.ask(r["text"], images=r.get("images", []), system=r.get("system"), max_new_tokens=int(r.get("max_new_tokens", 512)),
                            temperature=float(r.get("temperature", 0.0)), n=int(r.get("n", 1)), timeout_s=a.timeout)
            o["id"] = r["id"]
            f.write(json.dumps(o) + "\n"); f.flush()
            print(f"[{i + 1}/{len(reqs)}] {r['id']} {o['seconds']}s new={o['n_new_tokens']} finish={o.get('finish')}", flush=True)
    os.replace(part, out)
    Path(str(out) + ".DONE").write_text(f"{st.get('model')} {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
