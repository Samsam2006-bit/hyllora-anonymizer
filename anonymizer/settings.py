"""How a live camera can be run (the page's Läge, 2026-11-17, Sameh: "gör så man kan toggla så man kan se vilken som
funkar bäst"). Kept apart from main.py so the page's server reads them without loading torch.

Full is the analysis' own. Lätt is what 20 cameras on one GTX 1080 may need: the small model at 640 px, 5 frames a
second, no body vector, and no model on a camera where nothing moves. "fleet": all such cameras in one process
(fleet.py), the models loaded once and their frames through the model together."""

MODEL = "yolo26s-pose.pt"
IMGSZ = 960
PRESETS = {
    "full": {"model": "yolo26s-pose.pt", "imgsz": 960, "fps": 0, "look": True, "motion": False, "fleet": False, "hwdec": False},
    "latt": {"model": "yolo26n-pose.pt", "imgsz": 640, "fps": 5, "look": False, "motion": True, "fleet": True, "hwdec": False},
}
MODELS_OK = ("yolo26s-pose.pt", "yolo26n-pose.pt")
IMGSZ_OK = (640, 960)
FPS_MAX = 30.0


def check(model: str, imgsz: int, fps: float) -> str | None:
    """Why the settings cannot be used, or None."""
    if model not in MODELS_OK:
        return f"Modellen ska vara en av {', '.join(MODELS_OK)}"
    if imgsz not in IMGSZ_OK:
        return f"Bildstorleken ska vara {' eller '.join(map(str, IMGSZ_OK))}"
    if not 0 <= fps <= FPS_MAX:
        return f"Takten ska vara 0 (alla) till {FPS_MAX:.0f} bilder/s"
    return None
