"""Nothing in the analysis reaches the internet (REVIEW_BACKLOG S3): the weights are read from models/ with a
checksum (weights.py), and the libraries' own downloads, telemetry and package installs are switched off here,
before any of them is imported. Ultralytics otherwise sends an event to Google Analytics after each prediction run
and pip-installs a missing tracker dependency at run time; timm asks Hugging Face. scripts/weights.py fetches."""

import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # Ultralytics: no online checks, downloads or telemetry
os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # Ultralytics: never pip install at run time
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # timm: never ask Hugging Face
