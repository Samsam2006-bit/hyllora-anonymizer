"""The anonymizer (CURRENT_STATE J, decision 2026-10-13): the only program that sees the camera's picture. It finds
the people, their 17 keypoints and their tracks, and passes on only those numbers (a skeleton stream, skeleton.py)
with a picture of the empty store. No picture of a person leaves it, so the analysis (inferero) never sees one.

Never imports inferero.pipeline (as the hub does not): the anonymizer and the analysis only meet in the skeleton
stream, so the anonymizer can move to a box of its own or into the camera without the analysis changing.

Copyright (C) 2026 Hyllora. The anonymizer as it is given to a store (packaging/: this package, the inferero/ files
it imports and the build and install scripts, all in the installation as source) is free software under the GNU
Affero General Public License version 3 (anonymizer/LICENSE), as Ultralytics' YOLO that it runs on (REVIEW_BACKLOG
LIC1). It comes with ABSOLUTELY NO WARRANTY.
"""

import os

# RTSP video over TCP, not FFmpeg's default UDP: a Windows firewall lets the camera's answer in over the connection
# the anonymizer opened, but not the video on UDP ports of its own, and the camera then "cannot be opened"
# (ValueError at the store, 2026-11-13). Read by OpenCV each time a capture is opened; the camera processes inherit it.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

try:  # HTTPS (sending to the analysis) checked by the operating system, as the browser does (packaging/requirements.txt)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
