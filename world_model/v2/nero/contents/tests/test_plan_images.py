"""Plan images for the proposer (nero/contents/plan_images.py, 11-12): opt-in, default OFF.  No live call (mock backends / mock subprocess); renders on CPU (osmesa).
PYTHONPATH=. MUJOCO_GL=osmesa python -m pytest -q world_model/v2/nero/contents/tests/test_plan_images.py"""
import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from world_model.v2.house.lang.backend import Backend, CachedVLM
from world_model.v2.nero.contents import plan_images as PI
from world_model.v2.nero.pan import llm as LLM
from world_model.v2.stack.proposer import mock as MK, propose as PP, prompt as PR


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    d = tmp_path_factory.mktemp("img")
    a = PI.render_views("box_behind", 0, "zed+wrist", d / "a")
    b = PI.render_views("box_behind", 0, "zed+wrist", d / "b")
    return a, b


def test_render_shape_and_deterministic(views):
    a, b = views
    assert a.cameras == ["zed_left", "left_arm_wrist_rgb"] and len(a.paths) == 2 and a.note.startswith("IMAGES: image 1 is the torso stereo camera")
    import imageio.v2 as iio
    for p in a.paths:
        assert iio.imread(p).shape == (360, 640, 3)
    assert [os.path.basename(p) for p in a.paths] == [os.path.basename(p) for p in b.paths]          # same world -> same content-hash file names
    assert hashlib.sha256(open(a.paths[0], "rb").read()).hexdigest()[:16] == os.path.basename(a.paths[0])[:-4]
    assert a.assumptions["zed_pitch_deg"]["tag"] == "ASSUMED" and "PROPOSED" in a.assumptions["zed_mount"]


def test_mode_none_and_bad_mode(tmp_path):
    r = PI.render_views("box_behind", 0, "none", tmp_path)
    assert r.paths == [] and r.proposer_cfg() == {} and not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        PI.render_views("box_behind", 0, "stereo", tmp_path)


class Rec(Backend):
    name, model = "rec", "rec-model"

    def __init__(self):
        self.calls = []

    def ask(self, text, images=(), system=None, max_new_tokens=512, temperature=0.0, n=1):
        self.calls.append(dict(text=text, images=list(images)))
        return dict(texts=[json.dumps(MK.fixture_plan("box"))] * n, seconds=0.0, wall_s=0.0, cached=False, backend=self.name)


@pytest.fixture
def rec(monkeypatch):
    r = Rec()
    monkeypatch.setattr(PP, "make_backend", lambda spec, cache_dir, offline=False: CachedVLM(r, cache_dir, mode="ro" if offline else "rw"))
    return r


def _png(tmp_path, name, data=b"\x89PNGfake"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_flag_off_text_and_key_unchanged(rec, tmp_path):
    plan, r = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c"))
    assert rec.calls[0]["text"] == PR.user_prompt("box", PR.scene_facts("box", None)) and rec.calls[0]["images"] == [] and r["image"] is None
    old_key = CachedVLM(rec, str(tmp_path / "c2")).key(PR.user_prompt("box", PR.scene_facts("box", None)), [], PR.SYSTEM, PP.MAX_NEW_TOKENS, 0.0, 1)
    assert r["request_hash"] == old_key
    # images=[] / image_note given but no image: still text-only
    _, r2 = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c3"), images=[], image_note="IMAGES: x")
    assert r2["request_hash"] == old_key and rec.calls[1]["text"] == rec.calls[0]["text"]


def test_flag_on_key_differs_and_images_passed(rec, tmp_path):
    im = _png(tmp_path, "a.png")
    note = PI.image_note("zed")
    _, r0 = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c"))
    _, r1 = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c"), images=[im], image_note=note)
    assert r1["request_hash"] != r0["request_hash"] and rec.calls[1]["images"] == [im]
    assert rec.calls[1]["text"] == rec.calls[0]["text"] + "\n\n" + note and r1["image"] == [im]
    assert r1["raw_reply_path"].endswith(r1["request_hash"] + ".json") and not r1["cached"]
    # other bytes -> other key (a text-only cache entry is never replayed for an image request, and a changed image is a new request)
    im2 = _png(tmp_path, "b.png", b"\x89PNGother")
    _, r2 = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c"), images=[im2], image_note=note)
    assert r2["request_hash"] not in (r0["request_hash"], r1["request_hash"])
    # same images again -> cache hit
    _, r3 = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c"), images=[im], image_note=note)
    assert r3["request_hash"] == r1["request_hash"] and r3["cached"] and len(rec.calls) == 3
    # offline replay of a text-only cache never answers an image request
    PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "t"))
    with pytest.raises(PP.ProposerCacheMiss):
        PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "t"), offline=True, images=[im], image_note=note)


def test_reasks_carry_the_same_images(rec, tmp_path):
    im = _png(tmp_path, "a.png")
    note = PI.image_note("zed")
    plan, _ = PP.propose("box", backend_spec="fake:x", cache_dir=str(tmp_path / "c"), images=[im], image_note=note)
    n = len(rec.calls)
    ev = dict(trigger="T1", readings={}, probes_skipped=[], failing_rows=[])
    PP.reask("box", plan, ev, backend_spec="fake:x", cache_dir=str(tmp_path / "c"), images=[im], image_note=note)
    PP.reask_rejected("box", [dict(plan=plan, reply="{}", stage="schema", reasons=["x"])], backend_spec="fake:x", cache_dir=str(tmp_path / "c"), images=[im], image_note=note)
    assert [c["images"] for c in rec.calls[n:]] == [[im], [im]] and all(c["text"].endswith(note) for c in rec.calls[n:])


def test_probe_select_passes_images_from_proposer_cfg():
    import inspect
    from world_model.v2.nero.contents import probe_select as PS
    src = inspect.getsource(PS)
    assert 'images=cfg.get("images")' in src and 'image_note=cfg.get("image_note")' in src


def test_codex_backend_builds_data_url(monkeypatch, tmp_path):
    seen = {}

    class FakeCodex:
        def __init__(self, *a, **k):
            pass

        def __call__(self, body):
            seen.update(body=body)
            return "{}", "stop", {}

    import world_model.v2.agentchord_probe.gateway as GW
    monkeypatch.setattr(GW, "Codex", FakeCodex)
    im = _png(tmp_path, "a.png", b"abc")
    b = LLM.CodexVLM("gpt-5.6-sol", "high")
    b.ask("hello", images=[im], system="SYS")
    parts = seen["body"]["messages"][-1]["content"]
    assert parts[0] == dict(type="text", text="hello") and parts[1]["image_url"]["url"] == "data:image/png;base64,YWJj"
    # the real gateway turns that into codex `-i <file>` args
    prompt, imgs = GW.flatten(seen["body"]["messages"])
    assert len(imgs) == 1 and imgs[0].startswith("data:image/png;base64,")


def test_claude_backend_builds_image_block(monkeypatch, tmp_path):
    seen = {}
    lines = [dict(type="result", subtype="success", is_error=False, result="{}", stop_reason="end_turn")]

    def run(cmd, input=None, **kw):
        seen.update(cmd=cmd, input=input)
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(x) for x in lines), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    im = _png(tmp_path, "a.png", b"abc")
    LLM.make("claude:claude-sonnet-5-5:high").ask("hello", images=[im], system="SYS")
    blocks = json.loads(seen["input"])["message"]["content"]
    assert blocks[0]["type"] == "text" and blocks[0]["text"].endswith("hello")
    assert blocks[1] == dict(type="image", source=dict(type="base64", media_type="image/png", data="YWJj")) and "--tools" in seen["cmd"]
