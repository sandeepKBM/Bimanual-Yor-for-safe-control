"""grounding/gr_runner.py -- wraps vlm.runner (no change to it) with a compatibility patch so Molmo2-ER's remote-code processor loads:
its Molmo2Processor passes image_use_col_tokens & co. (optional_attributes) to ProcessorMixin.__init__, which the installed transformers
rejects ("Unexpected keyword argument image_use_col_tokens"). We pop those kwargs, call the stock __init__, then set them as attributes."""
import functools
import os

import transformers.processing_utils as pu

_OPT = {"image_use_col_tokens", "use_single_crop_col_tokens", "use_single_crop_start_token", "video_use_col_tokens",
        "use_frame_special_tokens", "time_mode"}
_orig = pu.ProcessorMixin.__init__


def _init(self, *a, **kw):
    extra = {k: kw.pop(k) for k in list(kw) if k in _OPT}
    _orig(self, *a, **kw)
    for k, v in extra.items():
        setattr(self, k, v)


pu.ProcessorMixin.__init__ = _init

# Molmo2 remote code looks up ROPE_INIT_FUNCTIONS["default"], which newer transformers no longer registers -> re-register the standard RoPE
import torch  # noqa: E402
from transformers import modeling_rope_utils as _mru  # noqa: E402

if "default" not in _mru.ROPE_INIT_FUNCTIONS:
    def _rope_default(config, device=None, seq_len=None, **kw):
        rp = getattr(config, "rope_parameters", None) or {}
        base = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv, 1.0
    _mru.ROPE_INIT_FUNCTIONS["default"] = _rope_default

# transformers 5.x renamed create_causal_mask(input_embeds=) -> inputs_embeds= (and dropped cache_position): adapt BEFORE the remote module imports them
import inspect  # noqa: E402
from transformers import masking_utils as _mk  # noqa: E402


def _compat(fn):
    params = set(inspect.signature(fn).parameters)

    def w(*a, **kw):
        if "input_embeds" in kw and "input_embeds" not in params and "inputs_embeds" in params:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        if "cache_position" in kw and "cache_position" not in params and "kwargs" not in params:
            kw.pop("cache_position")
        return fn(*a, **kw)
    return w


_mk.create_causal_mask = _compat(_mk.create_causal_mask)
_mk.create_masks_for_generate = _compat(_mk.create_masks_for_generate)

from world_model.v2.vlm import runner  # noqa: E402

# Molmo2's prepare_inputs_for_generation reads cache_position[0], which transformers 5.x generate() no longer passes -> synthesize it.
_orig_load = runner.load


def _load(key):
    proc, model, mid = _orig_load(key)
    if key == "molmo2-er":
        # non-persistent buffers (RoPE inv_freq) come out all-zero when the model is built on the meta device -> recompute them
        for name, mod in model.named_modules():
            if hasattr(mod, "rope_init_fn") and hasattr(mod, "inv_freq"):
                inv, sc = mod.rope_init_fn(mod.config, mod.inv_freq.device)
                mod.inv_freq = inv.to(mod.inv_freq.dtype)
                mod.original_inv_freq = mod.inv_freq
                mod.attention_scaling = sc
                print("recomputed inv_freq", name, float(inv[0]), float(inv[-1]), flush=True)
        for name, b in model.named_buffers():
            if b.numel() > 1 and float(b.float().abs().sum()) == 0.0:
                print("WARNING all-zero buffer", name, tuple(b.shape), flush=True)
        cls = type(model)
        orig = cls.prepare_inputs_for_generation

        @functools.wraps(orig)                       # generate() validates kwargs against this signature
        def pig(self, input_ids, *a, cache_position=None, past_key_values=None, **kw):
            if cache_position is None:
                pl = past_key_values.get_seq_length() if past_key_values is not None else 0
                n = input_ids.shape[1]
                cache_position = torch.arange(pl, n if n > pl else pl + n, device=input_ids.device)
            return orig(self, input_ids, *a, cache_position=cache_position, past_key_values=past_key_values, **kw)
        cls.prepare_inputs_for_generation = pig
        if os.environ.get("GR_MOLMO_NOCACHE"):
            model.generation_config.use_cache = False
    return proc, model, mid


runner.load = _load

if __name__ == "__main__":
    runner.main()
