"""Starts the installed anonymizer (ARCHITECTURE §13c). The Windows service (packaging/windows/service.xml) runs this
with the bundled Python, from the install folder:

    <install>/launcher.py   this file
    <install>/python/       Python with the packages (packaging/requirements.txt)
    <install>/app/          the program (packaging/stage.py): anonymizer/, the few inferero/ files it uses, static/
    <install>/models/       the weights it uses, checked by inferero/weights.py

It sets where the store's data is (%ProgramData%\\Hyllora\\Anonymizer, outside the program) and where the models are,
and runs the page on 127.0.0.1:8430 in this process. HYLLORA_INGEST_URL (the analysis' address) comes from the
service's environment (service.xml).
"""

import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
PORT = int(os.environ.get("HYLLORA_PORT") or 8430)  # another only to try it beside a running one


def data_dir() -> Path:
    pd = os.environ.get("PROGRAMDATA")
    return Path(pd) / "Hyllora" / "Anonymizer" if pd else HOME / "data"


def main() -> None:
    os.environ.setdefault("HYLLORA_DATA", str(data_dir()))
    os.environ.setdefault("HYLLORA_MODELS", str(HOME / "models"))
    Path(os.environ["HYLLORA_DATA"]).mkdir(parents=True, exist_ok=True)
    app = HOME / "app"
    os.chdir(app)  # the camera processes are started with cwd = the program (server.py)
    sys.path.insert(0, str(app))
    import uvicorn

    uvicorn.run("anonymizer.server:app", host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
