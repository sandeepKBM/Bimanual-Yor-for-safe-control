"""assets.py -- MolmoSpaces data paths, house extraction from the local val shard, texture-stripped XML copies,
object-pack presence check.  OFFLINE: reads only local files (no network here; the shard was fetched earlier by
HF byte-range reads, see docs/status/molmospaces_integration_plan.md).  Never imports tray_rep or any YOR driver.

Data root: env HOUSE_DATA (default data/molmospaces/assets).  Never /common/users or home.
License policy (plan section 1.1/6): code Apache-2.0; data odc-by / cc-by-4.0; THOR pack license NOT read; do not
redistribute assets or push them to GitHub.  Only THOR objects are used by procthor-10k houses.
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
from pathlib import Path

HOUSE_DATA = Path(os.environ.get("HOUSE_DATA", "data/molmospaces/assets"))
SCENE_DIR = HOUSE_DATA / "scenes" / "procthor-10k-val"
SHARD = HOUSE_DATA / "dl" / "val_shard0.tar"
THOR_DIR = HOUSE_DATA / "objects" / "thor"
LICENSE_NOTE = "MolmoSpaces data: odc-by/cc-by-4.0 (some Objaverse CC BY-NC; THOR pack license unread). Local research use only."

# the 20 val houses used for Phase 0 (fixed list, sorted)
VAL20 = list(range(20))


def house_xml(idx: int) -> Path:
    return SCENE_DIR / f"val_{idx}.xml"


def stripped_xml(idx: int) -> Path:
    return SCENE_DIR / f"stripped_val_{idx}.xml"


def extract_house(idx: int) -> Path:
    """Extract val_<idx> from the local shard tar (zstd members) into SCENE_DIR. Idempotent."""
    x = house_xml(idx)
    if x.exists():
        return x
    import zstandard
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"procthor-10k-val_val_{idx}.tar.zst"
    with tarfile.open(SHARD) as outer:
        raw = outer.extractfile(name).read()
    data = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read()
    with tarfile.open(fileobj=io.BytesIO(data)) as inner:
        inner.extractall(SCENE_DIR, filter="data")
    assert x.exists(), f"extraction of val_{idx} did not produce {x}"
    return x


def make_stripped(idx: int) -> Path:
    """Texture-stripped copy (textures, 0.64 GB, were not fetched; physics and rays ignore them). Idempotent."""
    out = stripped_xml(idx)
    if out.exists():
        return out
    s = house_xml(idx).read_text()
    s = re.sub(r"<texture [^>]*/>\s*", "", s)
    s = re.sub(r' texture="[^"]*"', "", s)
    out.write_text(s)
    return out


def missing_mesh_files(idx: int) -> list[str]:
    """Every mesh/file= path the house XML references that is absent on disk (empty list = all present)."""
    base = house_xml(idx).parent
    miss = []
    for f in sorted(set(re.findall(r'<mesh [^>]*file="([^"]+)"', house_xml(idx).read_text()))):
        if not (base / f).resolve().exists():
            miss.append(f)
    return miss


def metadata(idx: int) -> dict:
    return json.loads((SCENE_DIR / f"val_{idx}_metadata.json").read_text())["objects"]


def spec_json(idx: int) -> dict:
    return json.loads((SCENE_DIR / f"val_{idx}.json").read_text())


def door_table(idx: int) -> list[dict]:
    """Design door openings from the ProcTHOR spec: holePolygon width/height [m] (frame-to-frame cut in the wall)."""
    out = []
    for d in spec_json(idx).get("doors", []):
        h = d["holePolygon"]
        out.append(dict(id=d["id"], asset=d["assetId"], width=abs(h[1]["x"] - h[0]["x"]),
                        height=abs(h[1]["y"] - h[0]["y"]), room0=d.get("room0"), room1=d.get("room1")))
    return out


def disk_bytes(path: Path) -> int:
    tot = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def ensure_houses(ids=VAL20):
    for i in ids:
        extract_house(i)
        make_stripped(i)
    return [stripped_xml(i) for i in ids]
