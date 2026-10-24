"""LoRA fine-tuning of Qwen3.5-9B (vision-language) on the sim pick chat JSONL (ftdata/gen.py output).

TEMPLATE PARITY (the point of this file): the training input is
    [exact inference prompt token ids]  +  [answer ids + <|im_end|>]
where the prefix is produced by the SAME recipe as vlm/runner.py:run_one (system as a text-part list, user = one
{"type":"image"} placeholder per image + the text, apply_chat_template(add_generation_prompt=True, tokenize=False,
enable_thinking=False), then proc(text=[prompt], images=imgs)).  The answer is tokenized SEPARATELY and appended, so
the prefix is bit-identical to what the model sees at inference by construction (no BPE merge across the boundary).
Loss only on answer ids + <|im_end|>; everything else is -100.

  --dry-run   CPU only, processor/tokenizer only (no weights): token / visual-token / label-mask stats.
  train       needs a GPU; bf16 base, frozen vision tower, LoRA on the language-model projections, plain loop.
"""
import argparse, glob, json, math, os, random, re, signal, struct, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_DATA = "/filer/tmp1/ss5772/ft_data/sim_pick_v1/smoke"
DEFAULT_OUT_ROOT = "/filer/tmp1/ss5772/ft_runs"
END_TOKEN = "<|im_end|>"
DEFAULT_MAX_PIXELS = 640 * 480        # a 640x360 frame is untouched (-> ~225 visual tokens); larger frames are downscaled

# Qwen3.5 is hybrid: 24 linear-attention (Gated DeltaNet) layers + 8 full-attention layers (every 4th), all with a dense MLP.
# Verified against model.safetensors.index.json: full-attn has self_attn.{q,k,v,o}_proj, linear-attn has
# linear_attn.{in_proj_qkv,in_proj_z,out_proj} (in_proj_a/b are tiny per-head gates -> skipped), MLP has {gate,up,down}_proj.
# Anchored on model.language_model so the vision tower (model.visual.*) and the MTP head (mtp.*) are never touched.
# peft fullmatch()es a str target_modules against the module name.
LORA_TARGET_REGEX = (r"model\.language_model\.layers\.\d+\."
                     r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                     r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                     r"|mlp\.(gate_proj|up_proj|down_proj))")


# ----------------------------------------------------------------------------------------------- model dir / index
def model_dir(model_id=DEFAULT_MODEL):
    """Local snapshot dir of a cached HF model (no network)."""
    if os.path.isdir(model_id):
        return model_id
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = sorted(glob.glob(os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"{model_id} not in HF cache {hub}")
    return snaps[-1]


def lora_target_modules(mdir):
    """Module names (weight_map keys minus '.weight') matched by LORA_TARGET_REGEX; no weights loaded."""
    wm = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))["weight_map"]
    rx = re.compile(LORA_TARGET_REGEX)
    return sorted(k[:-len(".weight")] for k in wm if k.endswith(".weight") and rx.fullmatch(k[:-len(".weight")]))


def safetensors_shape(path, name):
    """Shape of one tensor from the safetensors header only (no weight loading)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))[name]["shape"]


def estimate_lora_params(mdir, r):
    wm = json.load(open(os.path.join(mdir, "model.safetensors.index.json")))["weight_map"]
    tot, cache = 0, {}
    for m in lora_target_modules(mdir):
        fn = wm[m + ".weight"]
        o, i = safetensors_shape(os.path.join(mdir, fn), m + ".weight")
        tot += r * (o + i)
    return tot


# ----------------------------------------------------------------------------------------------------- data / tokens
def load_jsonl(path, limit=0):
    recs = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for m in r["messages"]:
                if isinstance(m["content"], list):
                    for p in m["content"]:
                        if p.get("type") == "image" and not os.path.isabs(p["image"]):
                            p["image"] = os.path.join(base, p["image"])
            recs.append(r)
            if limit and len(recs) >= limit:
                break
    return recs


def split_record(rec):
    """-> (system str|None, user_text str, [image paths], answer str)."""
    sysm, text, imgs, ans = None, [], [], None
    for m in rec["messages"]:
        c = m["content"]
        if m["role"] == "system":
            sysm = c if isinstance(c, str) else "".join(p["text"] for p in c if p.get("type") == "text")
        elif m["role"] == "user":
            if isinstance(c, str):
                text.append(c)
            else:
                for p in c:
                    if p.get("type") == "image":
                        imgs.append(p["image"])
                    elif p.get("type") == "text":
                        text.append(p["text"])
        elif m["role"] == "assistant":
            ans = c if isinstance(c, str) else "".join(p["text"] for p in c if p.get("type") == "text")
    if ans is None:
        raise ValueError("record has no assistant message")
    return sysm, "\n".join(text) if len(text) != 1 else text[0], imgs, ans


def load_images(paths, max_pixels):
    from PIL import Image
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")           # same as runner.run_one
        if max_pixels and im.width * im.height > max_pixels:
            s = math.sqrt(max_pixels / (im.width * im.height))
            im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))), Image.LANCZOS)
        out.append(im)
    return out


def inference_prompt(proc, sysm, text, n_images):
    """EXACTLY runner.run_one's prompt construction (non-NO_SYSTEM branch)."""
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": text}]
    msgs = ([{"role": "system", "content": [{"type": "text", "text": sysm}]}] if sysm else []) + [{"role": "user", "content": content}]
    try:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def encode_example(proc, rec, max_pixels=DEFAULT_MAX_PIXELS):
    """-> dict(input_ids, labels (lists), pixel_values, image_grid_thw, extras{}, n_prompt, n_answer, n_visual, sample_id)."""
    import torch
    sysm, text, paths, ans = split_record(rec)
    imgs = load_images(paths, max_pixels)
    prompt = inference_prompt(proc, sysm, text, len(imgs))
    enc = proc(text=[prompt], images=imgs or None, return_tensors="pt")          # == runner.run_one (before .to(device))
    prefix = enc["input_ids"][0].tolist()
    tok = getattr(proc, "tokenizer", proc)
    ans_ids = tok(ans + END_TOKEN, add_special_tokens=False)["input_ids"]
    ids = prefix + ans_ids
    labels = [-100] * len(prefix) + ans_ids
    extras = {}
    for k, v in enc.items():
        if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            continue
        if hasattr(v, "shape") and v.ndim == 2 and v.shape[1] == len(prefix):      # per-token tensors (e.g. mm_token_type_ids)
            extras[k] = torch.cat([v[0], torch.zeros(len(ans_ids), dtype=v.dtype)])
        else:
            extras[k] = v
    img_tok = getattr(proc, "image_token_id", None)
    if img_tok is None:
        img_tok = tok.convert_tokens_to_ids("<|image_pad|>")
    return dict(input_ids=ids, labels=labels, pixel_values=enc.get("pixel_values"), image_grid_thw=enc.get("image_grid_thw"),
                extras=extras, n_prompt=len(prefix), n_answer=len(ans_ids), n_visual=sum(1 for t in prefix if t == img_tok),
                sample_id=rec.get("meta", {}).get("sample_id"), task=rec.get("meta", {}).get("task"))


class SampleStream:
    """Deterministic infinite sample stream: epoch e = random.Random(seed*1000003+e).shuffle(range(N)); a view over [start, stop)."""

    def __init__(self, proc, recs, seed, start, stop, max_pixels):
        self.proc, self.recs, self.seed, self.start, self.stop, self.mp = proc, recs, seed, start, stop, max_pixels
        self._perm = {}

    def index(self, g):
        n, e = len(self.recs), g // len(self.recs)
        if e not in self._perm:
            p = list(range(n))
            random.Random(self.seed * 1000003 + e).shuffle(p)
            self._perm[e] = p
        return self._perm[e][g % n]

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, i):
        return encode_example(self.proc, self.recs[self.index(self.start + i)], self.mp)


class ListData:
    def __init__(self, proc, recs, max_pixels):
        self.proc, self.recs, self.mp = proc, recs, max_pixels

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return encode_example(self.proc, self.recs[i], self.mp)


def make_collate(pad_id):
    import torch

    def collate(items):
        L = max(len(x["input_ids"]) for x in items)
        ids = torch.full((len(items), L), pad_id, dtype=torch.long)
        lab = torch.full((len(items), L), -100, dtype=torch.long)
        att = torch.zeros((len(items), L), dtype=torch.long)
        for b, x in enumerate(items):                     # right padding
            n = len(x["input_ids"])
            ids[b, :n] = torch.tensor(x["input_ids"]); lab[b, :n] = torch.tensor(x["labels"]); att[b, :n] = 1
        out = dict(input_ids=ids, labels=lab, attention_mask=att)
        pv = [x["pixel_values"] for x in items if x["pixel_values"] is not None]
        if pv:
            out["pixel_values"] = torch.cat(pv, 0)
            out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in items if x["image_grid_thw"] is not None], 0)
        for k in (items[0]["extras"] or {}):
            vs = [x["extras"][k] for x in items]
            if vs[0].ndim == 1 and vs[0].shape[0] == len(items[0]["input_ids"]):
                t = torch.zeros((len(items), L), dtype=vs[0].dtype)
                for b, v in enumerate(vs):
                    t[b, :v.shape[0]] = v
                out[k] = t
            else:
                out[k] = torch.cat(vs, 0)
        out["_n_label"] = int((lab != -100).sum())
        out["_n_tok"] = int(att.sum())
        return out
    return collate


# --------------------------------------------------------------------------------------------------------- dry run
def dry_run(a):
    proc = load_processor(a.model)
    tok = getattr(proc, "tokenizer", proc)
    mdir = model_dir(a.model)
    tm = lora_target_modules(mdir)
    print(f"model dir: {mdir}\nLoRA target modules matched in weight index: {len(tm)}  (e.g. {tm[0]}, {tm[-1]})")
    print(f"estimated LoRA params r={a.r}: {estimate_lora_params(mdir, a.r)/1e6:.1f}M")
    for split in (a.train, a.val):
        if not split or not os.path.exists(split):
            continue
        recs = load_jsonl(split, a.dry_run_n)
        st = dict(tot=[], prompt=[], ans=[], vis=[], lab_frac=[])
        by_task = {}
        for r in recs:
            for p in split_record(r)[2]:
                assert os.path.exists(p), p
            e = encode_example(proc, r, a.max_pixels)
            st["tot"].append(len(e["input_ids"])); st["prompt"].append(e["n_prompt"]); st["ans"].append(e["n_answer"])
            st["vis"].append(e["n_visual"]); st["lab_frac"].append(e["n_answer"] / len(e["input_ids"]))
            by_task.setdefault(e["task"], []).append(e["n_answer"])
        f = lambda v: f"min {min(v):.4g} mean {sum(v)/len(v):.4g} max {max(v):.4g}"
        print(f"\n== {split}  ({len(recs)} records)")
        print("  total tokens  :", f(st["tot"])); print("  prompt tokens :", f(st["prompt"]))
        print("  answer tokens :", f(st["ans"]), "(labelled, incl. <|im_end|>)"); print("  visual tokens :", f(st["vis"]))
        print("  labelled frac :", f(st["lab_frac"]))
        print("  answer tokens by task:", {k: f"n={len(v)} mean={sum(v)/len(v):.1f}" for k, v in by_task.items()})
    e = encode_example(proc, recs[0], a.max_pixels)
    lab = [t for t in e["labels"] if t != -100]
    print("\nsample labelled text:", repr(tok.decode(lab)[:300]))
    print("prompt tail        :", repr(tok.decode(e["input_ids"][e["n_prompt"] - 12:e["n_prompt"]])))


def load_processor(model_id):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_dir(model_id))


# --------------------------------------------------------------------------------------------------------- training
def atomic_save_ckpt(model, opt, sched, state, out, keep=2):
    import torch
    final = os.path.join(out, f"ckpt-{state['step']:06d}")
    tmp = final + ".tmp"
    if os.path.exists(tmp):
        import shutil; shutil.rmtree(tmp)
    os.makedirs(tmp)
    model.save_pretrained(tmp)
    torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), state=state), os.path.join(tmp, "trainer.pt"))
    if os.path.exists(final):
        import shutil; shutil.rmtree(final)
    os.replace(tmp, final)
    ck = sorted(glob.glob(os.path.join(out, "ckpt-??????")))
    for old in ck[:-keep]:
        import shutil; shutil.rmtree(old, ignore_errors=True)
    return final


def latest_ckpt(out):
    ck = sorted(d for d in glob.glob(os.path.join(out, "ckpt-??????")) if os.path.exists(os.path.join(d, "trainer.pt")))
    return ck[-1] if ck else None


def build_model(a):
    import torch
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(model_dir(a.model), dtype=torch.bfloat16, device_map="cuda")
    return wrap_lora(model, a)


def wrap_lora(model, a):
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)                       # includes the vision tower (stays frozen; LoRA never targets it)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", target_modules=LORA_TARGET_REGEX)
    model = get_peft_model(model, cfg)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ntot = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {ntr/1e6:.2f}M / {ntot/1e9:.3f}B ({100*ntr/ntot:.3f}%)", flush=True)
    return model, ntr


def to_dev(batch, dev):
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items() if not k.startswith("_")}


def evaluate(model, loader, dev):
    import torch
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            nl = b["_n_label"]
            loss = model(**to_dev(b, dev), use_cache=False).loss
            tot += float(loss) * nl; n += nl
    model.train()
    return tot / max(n, 1)


def train(a, model_factory=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    out = a.out or os.path.join(DEFAULT_OUT_ROOT, a.run_name)
    os.makedirs(out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    proc = load_processor(a.model)
    pad_id = getattr(proc, "tokenizer", proc).pad_token_id
    tr_recs, va_recs = load_jsonl(a.train), (load_jsonl(a.val) if a.val and os.path.exists(a.val) else [])
    if a.max_eval:
        va_recs = va_recs[:a.max_eval]
    N, B, A = len(tr_recs), a.batch_size, a.grad_accum
    steps = a.max_steps if a.max_steps > 0 else math.ceil(a.epochs * N / (B * A))
    warm = a.warmup_steps if a.warmup_steps >= 0 else max(1, int(0.05 * steps))
    model, ntr = (model_factory or build_model)(a)
    dev = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, warm, steps)
    state = dict(step=0)
    ck = None if a.no_resume else latest_ckpt(out)
    if ck:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(ck, "adapter_model.safetensors")))
        t = torch.load(os.path.join(ck, "trainer.pt"), map_location="cpu", weights_only=False)
        opt.load_state_dict(t["opt"]); sched.load_state_dict(t["sched"]); state = t["state"]
        print(f"resumed from {ck} at step {state['step']}", flush=True)
    cfg = dict(vars(a), out=out, n_train=N, n_val=len(va_recs), total_steps=steps, warmup=warm, trainable_params=ntr,
               lora_target_regex=LORA_TARGET_REGEX, resumed_from=ck, effective_batch=B * A,
               versions={m: __import__(m).__version__ for m in ("torch", "transformers", "peft")})
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=1)
    logf = open(os.path.join(out, "train_log.jsonl"), "a")

    def log(**kw):
        logf.write(json.dumps(kw) + "\n"); logf.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    collate = make_collate(pad_id)
    val_loader = DataLoader(ListData(proc, va_recs, a.max_pixels), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate) if va_recs else None
    start_s, stop_s = state["step"] * A * B, steps * A * B
    if start_s >= stop_s:
        print("already finished"); return
    loader = iter(DataLoader(SampleStream(proc, tr_recs, a.seed, start_s, stop_s, a.max_pixels), batch_size=B, shuffle=False,
                             num_workers=a.workers, collate_fn=collate, prefetch_factor=2 if a.workers else None))
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    while state["step"] < steps and not stop["flag"]:
        t0, ls, ntok, nl_tot = time.time(), 0.0, 0, 0
        for _ in range(A):
            b = next(loader)
            loss = model(**to_dev(b, dev), use_cache=False).loss
            (loss / A).backward()
            ls += float(loss.detach()) / A; ntok += b["_n_tok"]; nl_tot += b["_n_label"]
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
        lr = sched.get_last_lr()[0]
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        state["step"] += 1
        s = state["step"]
        if not math.isfinite(ls) or not math.isfinite(gn):
            log(step=s, loss=ls, grad_norm=gn, error="non-finite"); raise SystemExit("non-finite loss/grad")
        rec = dict(step=s, loss=ls, lr=lr, grad_norm=gn, label_tokens=nl_tot, tokens_per_s=ntok / (time.time() - t0),
                   gpu_mem_gb=(torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0), t=time.time())
        if val_loader is not None and (s % a.eval_every == 0 or s == steps):
            rec["eval_loss"] = evaluate(model, val_loader, dev)
        log(**rec)
        if s % a.save_every == 0 and s < steps:
            atomic_save_ckpt(model, opt, sched, state, out)
    atomic_save_ckpt(model, opt, sched, state, out)
    if state["step"] >= steps:                                   # final adapter in a stable place (atomic)
        tmp = os.path.join(out, "adapter.tmp"); fin = os.path.join(out, "adapter")
        import shutil
        shutil.rmtree(tmp, ignore_errors=True); model.save_pretrained(tmp)
        if os.path.exists(fin):                                  # NFS may leave .nfsXXXX stragglers -> move aside, never fail the swap
            os.replace(fin, f"{fin}.old.{int(time.time())}")
        os.replace(tmp, fin)
        for o in glob.glob(fin + ".old.*"):
            shutil.rmtree(o, ignore_errors=True)
        print(f"DONE adapter -> {fin}", flush=True)
    else:
        print(f"interrupted at step {state['step']}; checkpoint saved (rerun to resume)", flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=DEFAULT_DATA + "/train.jsonl")
    ap.add_argument("--val", default=DEFAULT_DATA + "/val.jsonl")
    ap.add_argument("--run-name", default=time.strftime("lora_%Y%m%d_%H%M%S"))
    ap.add_argument("--out", default=None, help="default /filer/tmp1/ss5772/ft_runs/<run-name>")
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=-1, help="-1 = 5%% of steps")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=1); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=25); ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--dry-run-n", type=int, default=16)
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse()
    dry_run(args) if args.dry_run else train(args)
