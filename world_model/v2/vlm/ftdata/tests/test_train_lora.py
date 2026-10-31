"""CPU tests for ftdata/train_lora.py -- processor/tokenizer only, NO model weights.
Run: PYTHONPATH=/common/users/ss5772/bimanual HF_HUB_OFFLINE=1 .../qwenft/bin/python -m pytest -q world_model/v2/vlm/ftdata/tests/test_train_lora.py
"""
import json, os, re, sys, types
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch = pytest.importorskip("torch")
# transformers binds the FLA (Triton, GPU-only) gated-delta kernels at import time if `fla` imports; these tests run the tiny model on CPU
# tensors (even on GPU hosts), so hide `fla` before the qwen3_5 modeling module is imported -> torch reference path.
sys.modules.setdefault("fla", None)
pytest.importorskip("transformers")
from world_model.v2.vlm.ftdata import train_lora as T
from world_model.v2.vlm import runner

DATA = os.environ.get("QWENFT_TEST_DATA", "/filer/tmp1/ss5772/ft_data/sim_pick_v1/smoke/train.jsonl")
pytestmark = pytest.mark.skipif(not os.path.exists(DATA), reason="smoke data missing")


@pytest.fixture(scope="module")
def proc():
    return T.load_processor(T.DEFAULT_MODEL)


@pytest.fixture(scope="module")
def recs():
    r = T.load_jsonl(DATA, 40)
    arms = [x for x in r if x["meta"]["task"] == "arms"][:3]
    name = [x for x in r if x["meta"]["task"] == "name"][:3]
    assert arms and name, "need both arms and name records in the first 40"
    return arms + name


def _runner_inputs(proc, rec):
    """Run the REAL runner.run_one with a fake model that records the tensors generate() receives."""
    sysm, text, paths, _ = T.split_record(rec)
    seen = {}

    class Fake:
        device = torch.device("cpu")

        def generate(self, **kw):
            seen.update(kw)
            return torch.cat([kw["input_ids"], torch.zeros(1, 1, dtype=torch.long)], 1)

    cuda = (torch.cuda.synchronize, torch.cuda.max_memory_allocated)
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    try:
        runner.run_one(proc, Fake(), dict(id="x", images=paths, text=text, system=sysm, max_new_tokens=4))
    finally:
        torch.cuda.synchronize, torch.cuda.max_memory_allocated = cuda
    return seen


def test_template_parity_with_real_runner(proc, recs):
    for rec in recs:
        e = T.encode_example(proc, rec)
        inf = _runner_inputs(proc, rec)
        ref = inf["input_ids"][0].tolist()
        assert e["input_ids"][:len(ref)] == ref                      # prefix identical to inference tokens
        assert e["n_prompt"] == len(ref)
        assert torch.equal(e["pixel_values"], inf["pixel_values"])
        assert torch.equal(e["image_grid_thw"], inf["image_grid_thw"])


def test_prompt_ends_with_nothink_generation_prefix(proc, recs):
    tok = proc.tokenizer
    e = T.encode_example(proc, recs[0])
    assert tok.decode(e["input_ids"][e["n_prompt"] - 8:e["n_prompt"]]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_label_mask_is_exactly_answer_plus_end(proc, recs):
    tok = proc.tokenizer
    for rec in recs:
        e = T.encode_example(proc, rec)
        ans = T.split_record(rec)[3]
        lab = [t for t in e["labels"] if t != -100]
        assert tok.decode(lab) == ans + T.END_TOKEN
        assert lab[-1] == tok.convert_tokens_to_ids("<|im_end|>") == tok.eos_token_id
        assert lab == e["input_ids"][e["n_prompt"]:]                  # labelled span is the contiguous tail
        assert all(l == -100 for l in e["labels"][:e["n_prompt"]])
        assert e["n_answer"] == len(lab)


def test_answer_tokens_match_free_tokenization(proc, recs):
    """Separate-tokenize == joint-tokenize at the boundary (so training continuation == what generation would emit)."""
    tok = proc.tokenizer
    for rec in recs:
        e = T.encode_example(proc, rec)
        tail = tok.decode(e["input_ids"][e["n_prompt"] - 6:e["n_prompt"]])
        joint = tok(tail + T.split_record(rec)[3] + T.END_TOKEN, add_special_tokens=False)["input_ids"]
        assert joint[-e["n_answer"]:] == e["input_ids"][e["n_prompt"]:]


def test_naming_records(proc, recs):
    names = [r for r in recs if r["meta"]["task"] == "name"]
    for rec in names:
        sysm, text, paths, ans = T.split_record(rec)
        assert text.startswith("Name the boxed object") and len(paths) == 1
        e = T.encode_example(proc, rec)
        assert 2 <= e["n_answer"] <= 12 and e["n_visual"] > 0


def test_image_paths_resolve_and_relative_paths(tmp_path):
    for r in T.load_jsonl(DATA, 40):
        for p in T.split_record(r)[2]:
            assert os.path.exists(p), p
    rel = tmp_path / "d.jsonl"
    rel.write_text(json.dumps({"messages": [{"role": "user", "content": [{"type": "image", "image": "im/a.png"}, {"type": "text", "text": "t"}]},
                                            {"role": "assistant", "content": "a"}]}) + "\n")
    assert T.load_jsonl(str(rel))[0]["messages"][0]["content"][0]["image"] == str(tmp_path / "im/a.png")


def test_visual_token_bound(proc, recs):
    for rec in recs:
        e = T.encode_example(proc, rec)
        assert 100 < e["n_visual"] <= 400                          # 640x360 -> a few hundred
    # a big frame is downscaled to the max_pixels cap (visual tokens ~ pixels / (32*32))
    from PIL import Image
    big = recs[0]; p = big["messages"][1]["content"][0]["image"]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        q = os.path.join(d, "big.png"); Image.new("RGB", (1920, 1080), (90, 120, 150)).save(q)
        rec = json.loads(json.dumps(big)); rec["messages"][1]["content"][0]["image"] = q
        cap = T.DEFAULT_MAX_PIXELS
        n_cap = T.encode_example(proc, rec, cap)["n_visual"]
        n_raw = T.encode_example(proc, rec, 0)["n_visual"]
        assert n_cap <= cap // (32 * 32) + 16 and n_raw > 1500


def test_collate_and_stream_determinism(proc, recs):
    col = T.make_collate(proc.tokenizer.pad_token_id)
    b = col([T.encode_example(proc, r) for r in recs[:2]])
    assert b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape
    assert b["pixel_values"].shape[0] == int(b["image_grid_thw"].prod(1).sum())
    assert int((b["labels"] != -100).sum()) == b["_n_label"]
    s1, s2 = T.SampleStream(proc, recs, 7, 0, 30, 0), T.SampleStream(proc, recs, 7, 0, 30, 0)
    assert [s1.index(i) for i in range(30)] == [s2.index(i) for i in range(30)]
    s3 = T.SampleStream(proc, recs, 7, 12, 30, 0)                    # resume view lines up with the full stream
    assert [s3.index(12 + i) for i in range(10)] == [s1.index(12 + i) for i in range(10)]
    assert sorted(s1.index(i) for i in range(len(recs))) == list(range(len(recs)))


def test_lora_targets_exist_in_weight_index():
    mdir = T.model_dir(T.DEFAULT_MODEL)
    tm = T.lora_target_modules(mdir)
    assert len(tm) == 8 * 4 + 24 * 3 + 32 * 3                         # 8 full-attn (q,k,v,o) + 24 linear-attn (qkv,z,out) + 32 MLPs
    assert not any("visual" in m or m.startswith("mtp") for m in tm)
    assert all(m.startswith("model.language_model.layers.") for m in tm)
    assert {m.rsplit(".", 1)[1] for m in tm} >= {"q_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "down_proj"}
    assert 20e6 < T.estimate_lora_params(mdir, 16) < 200e6


def test_runner_registers_lora_key_and_default_unchanged():
    assert runner.MODELS["qwen3.5-9b-lora"] == "Qwen/Qwen3.5-9B"
    assert runner.MODELS["qwen3.5-9b"] == "Qwen/Qwen3.5-9B"
    src = open(runner.__file__).read()
    assert 'QWEN_LORA_PATH' in src and 'model_key == "qwen3.5-9b-lora"' in src
    assert "qwen3.5-9b-lora" not in runner.NO_SYSTEM | runner.OPENAI | runner.REMOTE_CODE
    assert runner.main.__defaults__ is None and "qwen3.5-9b" in src    # default --model arg untouched


def test_dry_run_cli(capsys):
    a = T.parse(["--dry-run", "--dry-run-n", "4"])
    T.dry_run(a)
    out = capsys.readouterr().out
    assert "LoRA target modules matched" in out and "visual tokens" in out and "labelled frac" in out


# ---- plumbing test on a TINY randomly-initialised Qwen3_5 (same architecture, ~20M params; no real weights) -------------
def _tiny_model():
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(T.model_dir(T.DEFAULT_MODEL))
    t, v = cfg.text_config, cfg.vision_config
    t.num_hidden_layers = 4; t.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    t.hidden_size = 64; t.intermediate_size = 128
    for k, val in dict(num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                       linear_key_head_dim=16, linear_value_head_dim=16).items():
        if hasattr(t, k):
            setattr(t, k, val)
    v.depth = 2; v.hidden_size = 64; v.intermediate_size = 128; v.num_heads = 4; v.out_hidden_size = 64
    if hasattr(v, "deepstack_visual_indexes"):
        v.deepstack_visual_indexes = []
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32)


def test_tiny_model_lora_loop_resume(proc, tmp_path):
    a = T.parse(["--out", str(tmp_path), "--max-steps", "3", "--grad-accum", "2", "--eval-every", "2", "--save-every", "2",
                 "--workers", "0", "--max-eval", "2", "--lr", "1e-3"])
    factory = lambda args: T.wrap_lora(_tiny_model(), args)
    T.train(a, model_factory=factory)
    lines = [json.loads(l) for l in open(tmp_path / "train_log.jsonl")]
    assert [l["step"] for l in lines] == [1, 2, 3]
    assert all(set(l) >= {"loss", "lr", "grad_norm", "tokens_per_s", "gpu_mem_gb"} for l in lines)
    assert "eval_loss" in lines[1] and "eval_loss" in lines[2] and "eval_loss" not in lines[0]
    assert lines[0]["lr"] < lines[1]["lr"] or lines[0]["lr"] > 0
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists() and (tmp_path / "run_config.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    # LoRA touched only the language model; vision tower untouched
    from safetensors.torch import load_file
    keys = load_file(str(tmp_path / "adapter" / "adapter_model.safetensors")).keys()
    assert keys and all("language_model" in k for k in keys) and not any("visual" in k for k in keys)
    # resume from ckpt-000003 is a no-op; resume from an earlier step continues at 3
    import shutil
    [shutil.rmtree(p, ignore_errors=True) for p in tmp_path.glob("ckpt-000003")]
    assert (tmp_path / "ckpt-000002").exists()
    T.train(a, model_factory=factory)
    lines = [json.loads(l) for l in open(tmp_path / "train_log.jsonl")]
    assert [l["step"] for l in lines] == [1, 2, 3, 3]
