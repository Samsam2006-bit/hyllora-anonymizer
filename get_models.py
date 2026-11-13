"""The three model weights the anonymizer uses, into models/ with their published checksums (start.bat runs it)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "packaging"))

import stage  # noqa: E402

stage.fetch_models(ROOT / "models")
print("Modellerna finns i models/")
