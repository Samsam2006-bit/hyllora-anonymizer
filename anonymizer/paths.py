"""Where the anonymizer keeps what it makes and what the store sets up. In the repo it is data/ as for the analysis;
installed (packaging/, a Windows service) the launcher sets HYLLORA_DATA to a folder outside the program
(C:\\ProgramData\\Hyllora\\Anonymizer), so an update that swaps the program never touches the store's keys,
streams or photos. Read once at import: the launcher sets it before anything is imported, and the camera processes
inherit it."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HYLLORA_DATA") or ROOT / "data")
UPLOADS = DATA / "uploads"  # stands in for the cameras in the test phase
SKELETONS = DATA / "skeletons"
OBJECTS = DATA / "objects"
OWN = DATA / "anonymizer"  # the box's keys (ingest_key, view_token)
