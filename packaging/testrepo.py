"""Copies the anonymizer to its public test repo (github.com/Samsam2006-bit/hyllora-anonymizer, AGPL), where a
Windows computer downloads it as a ZIP and starts it with start.bat (packaging/testrepo/). Only what stage.FILES
lists goes, never data/ or models/, and not the Google Sans fonts (not ours to publish: the page falls back).

    .venv/bin/python packaging/testrepo.py <clone of the test repo>          # copy, show what changed
    .venv/bin/python packaging/testrepo.py <clone of the test repo> --push   # and commit and push it
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import stage  # noqa: E402

EXTRA = {"start.bat": "start.bat", "get_models.py": "get_models.py", "README.md": "README.md",
         "gitignore": ".gitignore"}


def files() -> list[str]:
    return [n for n in stage.files() if not n.startswith("static/vendor/fonts/")]


def export(dst: Path) -> None:
    if not (dst / ".git").is_dir():
        sys.exit(f"{dst} är ingen git-klon av testrepot")
    for p in dst.iterdir():  # what is gone here goes there too
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for n in files():
        (dst / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / n, dst / n)
    for src, name in EXTRA.items():
        shutil.copyfile(HERE / "testrepo" / src, dst / name)
    # cmd misreads labels and goto in a .bat with Unix line ends: CRLF, kept as is by git (-text)
    bat = dst / "start.bat"
    bat.write_bytes(bat.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    (dst / ".gitattributes").write_text("*.bat -text\n")
    shutil.copyfile(ROOT / "anonymizer" / "LICENSE", dst / "LICENSE")
    fonts = dst / "static" / "vendor" / "fonts"  # the page mounts this folder: it must exist
    fonts.mkdir(parents=True, exist_ok=True)
    (fonts / "README.md").write_text("Hylloras typsnitt ingår inte här; sidan använder systemets.\n")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dst = Path(sys.argv[1]).resolve()
    export(dst)
    git = lambda *a: subprocess.run(["git", "-C", str(dst), *a], check=True, capture_output=True, text=True).stdout
    git("add", "-A")
    print(git("status", "--short") or "Inget ändrat")
    if "--push" in sys.argv and git("status", "--porcelain"):
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        git("commit", "-m", f"Från inferero {sha}")
        git("push", "-u", "origin", "HEAD")
        print("Pushat")


if __name__ == "__main__":
    main()
