from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import random
import csv

app = FastAPI()

# ============================================================
# FOLDER SETUP
# ============================================================
BASE = Path(__file__).parent

TEMPLATES = BASE / "templates"
STIMULI = BASE / "stimuli"
CLEAN_DIR = STIMULI / "clean"
ADV_DIR = STIMULI / "adv"
RESULTS_FILE = STIMULI / "results.csv"

# Mount static files (for images only)
app.mount("/stimuli", StaticFiles(directory=STIMULI), name="stimuli")

# Allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ============================================================
# DOG LABELS
# ============================================================
BREEDS = [
    "Shih-Tzu",
    "Rhodesian ridgeback",
    "Beagle",
    "English foxhound",
    "Border terrier",
    "Australian terrier",
    "Golden retriever",
    "Old English sheepdog",
    "Samoyed",
    "Dingo",
]

LABEL_MAP = {
    "n02086240": "Shih-Tzu",
    "n02087394": "Rhodesian ridgeback",
    "n02088364": "Beagle",
    "n02089973": "English foxhound",
    "n02093754": "Border terrier",
    "n02096294": "Australian terrier",
    "n02099601": "Golden retriever",
    "n02105641": "Old English sheepdog",
    "n02111889": "Samoyed",
    "n02115641": "Dingo",
}

# ============================================================
# LOAD 100 BASE IMAGES FROM CLEAN (backbone)
# ============================================================

BASE_IMAGES = []

for wnid_dir in CLEAN_DIR.iterdir():
    if wnid_dir.is_dir():
        wnid = wnid_dir.name
        for img in wnid_dir.iterdir():
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                BASE_IMAGES.append((wnid, img.name))

if len(BASE_IMAGES) != 100:
    print(f"❌ WARNING: expected 100 clean images, found {len(BASE_IMAGES)}")

# ============================================================
# TRIAL GENERATION
# ============================================================

MODEL_NAMES = [
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384",
    "convnext_tiny",
    "resnet50",
    "resnet18",
    "densenet121",
]

SESSIONS = {}


def build_trials():
    """Builds a 100-trial session: 25 clean + 75 adv (15 per model)."""
    if len(BASE_IMAGES) < 100:
        raise RuntimeError("Not enough base images loaded!")

    indices = list(range(100))
    random.shuffle(indices)

    clean_idx = indices[:25]
    adv_idx = indices[25:100]

    trials = []

    # 25 clean
    for idx in clean_idx:
        wnid, fname = BASE_IMAGES[idx]
        trials.append({
            "img_url": f"stimuli/clean/{wnid}/{fname}",
            "true_label": LABEL_MAP[wnid],
            "condition": "clean",
            "model": "none",
            "target_label": None
        })

    # 75 adversarial (15 from each of 5 models)
    pointer = 0
    for model in MODEL_NAMES:
        for _ in range(15):
            wnid, fname = BASE_IMAGES[adv_idx[pointer]]
            pointer += 1
            trials.append({
                "img_url": f"stimuli/adv/{model}/{wnid}/{fname}",
                "true_label": LABEL_MAP[wnid],
                "condition": "adv",
                "model": model,
                "target_label": None
            })

    random.shuffle(trials)

    # assign trial IDs
    for i, t in enumerate(trials):
        t["trial_id"] = i

    return trials


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    """Load study.html from templates."""
    return FileResponse(TEMPLATES / "study.html")


@app.get("/study")
def study():
    """Alias for convenience."""
    return FileResponse(TEMPLATES / "study.html")


@app.get("/trial")
def get_trial(participant_name: str):
    """Returns next trial for the participant."""
    if participant_name not in SESSIONS:
        SESSIONS[participant_name] = build_trials()

    trials = SESSIONS[participant_name]

    if len(trials) == 0:
        return {"done": True}

    trial = trials.pop(0)

    # Fixed (non-shuffled) order of all 10 options
    trial["options"] = BREEDS

    return trial


class Resp(BaseModel):
    participant_name: str
    trial_id: int
    model: str
    condition: str
    response: str
    true_label: str
    target_label: str | None = None
    img_url: str


@app.post("/response")
def save_response(resp: Resp):
    """Save response into CSV."""
    first = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow([
                "participant_name", "trial_id", "model",
                "condition", "response", "true_label",
                "target_label", "img_url"
            ])
        writer.writerow([
            resp.participant_name, resp.trial_id, resp.model,
            resp.condition, resp.response, resp.true_label,
            resp.target_label, resp.img_url
        ])
    return {"status": "ok"}