"""Lays out the anonymizer as it is installed (packaging/launcher.py): the program in <out>/app, the weights it uses
in <out>/models and the launcher in <out>. Runs on any OS; packaging/windows/build.ps1 calls it.

    python packaging/stage.py <out>

The weights are copied from models/ when they are there with the published checksum, else fetched and checked.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inferero.weights import WEIGHTS, sha256  # noqa: E402

# what the anonymizer runs on: its own package, the few analysis files it imports (never inferero.pipeline), and the
# page's look. A new import from inferero/ must be added here, or the installed app fails to start
# (tests/test_packaging.py checks it).
FILES = ["anonymizer/**/*.py", "anonymizer/static/**/*", "inferero/__init__.py", "inferero/weights.py",
         "inferero/reid.py", "inferero/gpu.py", "inferero/botsort.yaml", "static/tokens.css", "static/style.css",
         "static/hyllora.svg", "static/vendor/fonts/**/*", "anonymizer/LICENSE",
         # AGPL: the scripts that build and install it are part of its source, so they go with it
         "packaging/*.py", "packaging/requirements.txt", "packaging/windows/*.ps1", "packaging/windows/*.iss",
         "packaging/windows/*.xml"]
# the person and pose model (main.MODEL), the body vector (reid.REID_MODEL), the search for baskets (objects.MODEL)
MODELS = ["yolo26s-pose.pt", "vit_pe_core_base_patch16_224.fb.safetensors", "yoloe-26m-seg.pt"]


def files() -> list[str]:
    out = []
    for pattern in FILES:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("."):
                out.append(p.relative_to(ROOT).as_posix())
    return out


def stage(out: Path) -> None:
    app = out / "app"
    shutil.rmtree(app, ignore_errors=True)
    for n in files():
        (app / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / n, app / n)
    shutil.copyfile(ROOT / "packaging" / "launcher.py", out / "launcher.py")
    fetch_models(out / "models")
    print(f"{out}: {len(files())} filer i app/, {len(MODELS)} modeller")


def fetch_models(models: Path) -> None:
    """MODELS in `models`, each with the published checksum (also packaging/testrepo/get_models.py)."""
    try:  # HTTPS checked by the operating system: Python's own check fails on a fresh Windows (requirements.txt)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    models.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        want, url = WEIGHTS[name]
        dst = models / name
        if dst.is_file() and sha256(dst) == want:
            continue
        src = ROOT / "models" / name
        part = dst.with_name(name + ".part")
        if src.is_file() and sha256(src) == want:
            shutil.copyfile(src, part)
        else:
            print(f"Hämtar {name} …")
            urllib.request.urlretrieve(url, part)
        if sha256(part) != want:
            part.unlink()
            sys.exit(f"{name}: fel kontrollsumma")
        part.replace(dst)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    stage(Path(sys.argv[1]))
