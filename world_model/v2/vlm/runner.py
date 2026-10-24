"""vlm/runner.py -- batch VLM runner (runs INSIDE a Slurm job on ilab3, `vlm` conda env). No server, no network at inference.

Request JSONL (one object per line):
  {"id": str, "images": [path,...], "text": str, "system": str|null, "max_new_tokens": 512, "temperature": 0.0, "n": 1,
   "top_p": 0.95}          temperature 0 -> greedy (n forced to 1); temperature > 0 -> n independent samples
Output JSONL: {"id", "texts": [n strings], "n_prompt_tokens", "n_new_tokens": [per sample], "seconds", "peak_vram_gb"}
Sentinels (NFS): <out>.DONE on success (written last, after the atomic rename), <out>.FAIL (traceback text) on error.
Resumes: ids already present in <out>.partial are skipped.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path

MODELS = {"qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3.5-9b-lora": "Qwen/Qwen3.5-9B", "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
          "cosmos-reason1-7b": "nvidia/Cosmos-Reason1-7B", "molmo2-er": "allenai/Molmo2-ER",
          "robobrain2.5-4b": "BAAI/RoboBrain2.5-4B", "robobrain2.0-7b": "BAAI/RoboBrain2.0-7B",
          "cosmos-reason2-8b": "nvidia/Cosmos-Reason2-8B", "cosmos3-edge": "nvidia/Cosmos3-Edge", "cosmos3-nano": "nvidia/Cosmos3-Nano"}
EXTRA_TOKENS = {"cosmos-reason1-7b": 700, "cosmos-reason2-8b": 700,   # models that emit a <think> block before the answer
                "cosmos3-edge": 4096, "cosmos3-nano": 4096}
OPENAI = {"cosmos3-edge", "cosmos3-nano"}                            # served by a local vLLM (vlm/slurm/cosmos3_serve.sh), not HF
NO_SYSTEM = {"molmo2-er"}                                           # chat template has no system role: system text is prepended to the user text
REMOTE_CODE = {"molmo2-er"}                                         # repo ships modeling code (read before use; job is offline)


def load(model_key):
    mid = MODELS.get(model_key, model_key)
    if model_key in OPENAI:
        from world_model.v2.vlm.openai_backend import OpenAIChat
        return None, OpenAIChat(mid), mid
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    rc =model_key in REMOTE_CODE
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=rc)
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=rc)
    if model_key == "qwen3.5-9b-lora":     # base Qwen3.5-9B + PEFT adapter (ftdata/train_lora.py output) from $QWEN_LORA_PATH; not merged
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, os.environ["QWEN_LORA_PATH"])
    model.eval()
    return proc, model, mid


def run_one(proc, model, r, key="qwen3.5-9b"):
    if key in OPENAI:
        return model.run_one(r, int(r.get("max_new_tokens", 512)) + EXTRA_TOKENS.get(key, 0))
    import torch
    from PIL import Image
    imgs = [Image.open(p).convert("RGB") for p in r.get("images", [])]
    text, sysm = r["text"], r.get("system")
    if key in NO_SYSTEM and sysm:
        text, sysm = sysm + "\n\n" + text, None
    if key in NO_SYSTEM:      # Molmo-style: images passed inside the message, template tokenizes
        content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]
        inp = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt")
        inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
    else:
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]
        msgs = ([{"role": "system", "content": [{"type": "text", "text": sysm}]}] if sysm else []) + [{"role": "user", "content": content}]
        try:
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = proc(text=[prompt], images=imgs or None, return_tensors="pt").to(model.device)
    temp = float(r.get("temperature", 0.0))
    n = int(r.get("n", 1)) if temp > 0 else 1
    gk = dict(max_new_tokens=int(r.get("max_new_tokens", 512)) + EXTRA_TOKENS.get(key, 0))
    if temp > 0:
        gk.update(do_sample=True, temperature=temp, top_p=float(r.get("top_p", 0.95)))
    else:
        gk.update(do_sample=False)
    tok = getattr(proc, "tokenizer", proc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # NB: num_return_sequences with multimodal inputs produced mis-aligned image features for samples k>0 in a first run
    # (model claimed "heavily corrupted image" in ~35% of samples). Instead: n identical prompts as a proper batch (or n calls).
    if n > 1 and key not in NO_SYSTEM:
        inp = proc(text=[prompt] * n, images=(imgs * n) or None, return_tensors="pt", padding=True).to(model.device)
        calls = [inp]
    else:
        calls = [inp] * n
    torch.cuda.synchronize(); t0 = time.time()
    gens = []
    with torch.no_grad():
        for c in calls:
            out = model.generate(**c, **gk)
            gens += list(out[:, c["input_ids"].shape[1]:])
    torch.cuda.synchronize(); dt = time.time() - t0
    L = calls[0]["input_ids"].shape[1]
    nnew = [int((g != pad).sum()) for g in gens]
    texts = tok.batch_decode(gens, skip_special_tokens=True)
    return dict(id=r["id"], texts=texts, n_prompt_tokens=int(L), n_new_tokens=nnew, seconds=round(dt, 3),
                peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5-9b")
    a = ap.parse_args()
    out = Path(a.out)
    part = Path(str(out) + ".partial")
    try:
        t0 = time.time()
        proc, model, mid = load(a.model)
        print(f"loaded {mid} in {time.time()-t0:.1f}s", flush=True)
        reqs = [json.loads(l) for l in open(a.req) if l.strip()]
        done = set()
        if part.exists():
            done = {json.loads(l)["id"] for l in open(part) if l.strip()}
        with open(part, "a") as f:
            for i, r in enumerate(reqs):
                if r["id"] in done:
                    continue
                o = run_one(proc, model, r, a.model)
                f.write(json.dumps(o) + "\n"); f.flush()
                print(f"[{i+1}/{len(reqs)}] {r['id']} {o['seconds']}s new={o['n_new_tokens']} vram={o['peak_vram_gb']}GB", flush=True)
        os.replace(part, out)
        Path(str(out) + ".DONE").write_text(f"{mid} {time.time()-t0:.1f}s\n")
    except BaseException:
        Path(str(out) + ".FAIL").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
