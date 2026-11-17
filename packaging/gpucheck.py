"""What start.bat (packaging/testrepo/run.ps1) says about the graphics card: its name, and whether torch can really
run on it. A card that torch sees but has no kernels for (a GTX 1080 with PyTorch's CUDA 12.8 build, 2026-11-17)
looked fine here before and failed only in the cameras. Plain ASCII: the window is the old Windows console."""
import sys

import torch

cuda = sys.argv[1] if len(sys.argv) > 1 else "?"
if not torch.cuda.is_available():
    print("Grafikkort: INGET, kor pa processorn. Uppdatera NVIDIA-drivrutinen.")
    sys.exit(0)
name, (major, minor) = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
try:
    ok = float((torch.ones(64, device="cuda") @ torch.ones(64, device="cuda")).item()) == 64.0
    why = ""
except Exception as e:  # "no kernel image is available for execution on the device"
    ok, why = False, str(e).splitlines()[0][:160]
half = "16-bit" if major >= 7 else "32-bit (16-bit ar langsamt pa det har kortet)"
print(f"Grafikkort: {name} (compute {major}.{minor}, torch {torch.__version__}, {cuda}), {half}")
if not ok:
    print(f"VARNING: torch kan inte rakna pa grafikkortet: {why}")
