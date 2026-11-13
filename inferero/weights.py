"""The model weights, read only from models/ and only if their checksum is the published one (REVIEW_BACKLOG S3).

Nothing is downloaded at run time: a store without internet must run, and a swapped file must not. The files are
put in models/ by scripts/weights.py (copied with the app to a store's computer, or fetched once). The sha256 of
each Ultralytics file is the digest GitHub publishes for the v8.4.0 release asset; PE-core's and Depth Anything's
are the LFS oid on Hugging Face at a fixed revision. A file is checked once per process, the first time it is loaded.
"""

import hashlib
import os
import threading
from pathlib import Path

# installed (packaging/), the launcher points HYLLORA_MODELS at the models beside the program, kept across updates
DIR = Path(os.environ.get("HYLLORA_MODELS") or Path(__file__).resolve().parent.parent / "models")

_GH = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
_HF = "https://huggingface.co/timm/vit_pe_core_base_patch16_224.fb/resolve/0f9fb1eea04d135faa1f15af05734fb996fcaf94/"
_DA = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f0f36572d16453bbda7a8389317b1bef99/"
WEIGHTS = {  # file in models/ -> (sha256, where scripts/weights.py fetches it)
    "yolo26s-pose.pt": ("a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646", _GH + "yolo26s-pose.pt"),
    "yolo26m-pose.pt": ("2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4", _GH + "yolo26m-pose.pt"),
    "yolo26n-pose.pt": ("eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9", _GH + "yolo26n-pose.pt"),
    "yolo26s.pt": ("646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b", _GH + "yolo26s.pt"),
    "yolo26m.pt": ("401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7", _GH + "yolo26m.pt"),
    # the anonymizer's search for what the store shows it a photo of (anonymizer/objects.py)
    "yoloe-26m-seg.pt": ("585f5ec9028fd358035da8d860c27c56be285a795cba2076fba536a4391c2c83", _GH + "yoloe-26m-seg.pt"),
    "sam2.1_b.pt": ("f1a9cf2dd69d84bb463b5ad98246d03e2d47a130a9295db0ec967e6cd95e2e47", _GH + "sam2.1_b.pt"),
    "vit_pe_core_base_patch16_224.fb.safetensors": (
        "802a8c86fed61c32573767f5b64c2476151fdb5c36f1ec0f2c9d0540c075530c", _HF + "model.safetensors"),
    # depth from one picture, where the store has changed since its 3D scan (scenefix.py; Apache-2.0)
    "depth_anything_v2_small.safetensors": (
        "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70", _DA + "model.safetensors"),
    "depth_anything_v2_small.config.json": (
        "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75", _DA + "config.json"),
}

_checked: set[str] = set()
_lock = threading.Lock()


def sha256(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path(name: str) -> str:
    """The checked local file for a weight name (e.g. "yolo26s-pose.pt"). Raises FileNotFoundError when it is not in
    models/, ValueError when it is not one of WEIGHTS or its content is not the published one."""
    if name not in WEIGHTS:
        raise ValueError(f"Okänd modellvikt: {name}")
    file = DIR / name
    with _lock:
        if name not in _checked:
            if not file.is_file():
                raise FileNotFoundError(f"Modellvikten {name} saknas i models/. Kör: .venv/bin/python scripts/weights.py")
            if sha256(file) != WEIGHTS[name][0]:
                raise ValueError(f"Modellvikten {name} har fel kontrollsumma. Hämta den igen med scripts/weights.py")
            _checked.add(name)
    return str(file)
