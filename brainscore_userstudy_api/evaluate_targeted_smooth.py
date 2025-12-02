import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import pandas as pd

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATA_DIR = Path("/scratch/kimias/brainscore_project/imagewoof_sample_100")
OUT_DIR = Path("/scratch/kimias/brainscore_project/results_targeted_smooth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MODEL_NAMES = [
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384",
    "convnext_tiny",
    "resnet50",
    "resnet18",
    "densenet121",
]

# ---- Targeted APGD hyperparams ----
EPSILON = 12 / 255         # slightly bigger for visibility
ALPHA = 3 / 255            # scaled step size
STEPS = 30                 # more steps, smoother gradients

LOWFREQ_SMOOTH = 5         # Gaussian smoothing kernel size

# ------------------------------------------------------------
# ImageNet → readable names
# ------------------------------------------------------------
from imagenet_labels import IMAGENET_CLASS_INDEX
LABELS = IMAGENET_CLASS_INDEX


# ------------------------------------------------------------
# ImageWoof mapping
# ------------------------------------------------------------
IMAGEWOOF_LABELS = {
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
IMAGEWOOF_CLASSNAMES = set(IMAGEWOOF_LABELS.values())


# ------------------------------------------------------------
# Preprocess: spatial transforms ONLY
# ------------------------------------------------------------
def build_preprocess(model):
    cfg = timm.data.resolve_data_config({}, model=model)
    _, H, W = cfg["input_size"]

    transform = transforms.Compose([
        transforms.Resize(H),
        transforms.CenterCrop((H, W)),
        transforms.ToTensor()
    ])

    return transform, cfg["mean"], cfg["std"]


# ------------------------------------------------------------
# Readable class name
# ------------------------------------------------------------
def decode_pred(idx: int):
    raw = LABELS[idx]
    return raw.split(",")[0].strip()

# ------------------------------------------------------------
# Smooth the gradient / perturbation (low-frequency)
# ------------------------------------------------------------
def smooth_tensor(tensor, kernel_size=5):
    """Apply Gaussian blur to smooth noise."""
    if kernel_size <= 1:
        return tensor

    t = tensor
    padding = kernel_size // 2
    kernel = torch.ones((3,1,kernel_size,kernel_size), device=t.device)
    kernel = kernel / kernel.sum()

    smoothed = torch.nn.functional.conv2d(
        t, kernel, padding=padding, groups=3
    )
    return smoothed


# ------------------------------------------------------------
# Target selection
# ------------------------------------------------------------
def pick_target_class(model, x, mean, std, true_label):
    with torch.no_grad():
        logits = model((x - mean) / std)
        probs = torch.softmax(logits, dim=1)[0]

    sorted_idx = torch.argsort(probs, descending=True)

    # Prefer ImageWoof classes
    for idx in sorted_idx:
        idx = idx.item()
        name = decode_pred(idx)
        if name in IMAGEWOOF_CLASSNAMES and name != true_label:
            return idx, name

    # fallback
    for idx in sorted_idx:
        idx2 = idx.item()
        name = decode_pred(idx2)
        if name != true_label:
            return idx2, name

    return None, None


# ------------------------------------------------------------
# Targeted PGD with LOW-FREQ smoothing
# ------------------------------------------------------------
def targeted_pgd_smooth(model, x, mean, std, target_idx):
    x_orig = x.clone().detach()
    x_adv = x_orig.clone().detach()

    for _ in range(STEPS):
        x_adv.requires_grad_(True)
        logits = model((x_adv - mean) / std)

        target_tensor = torch.tensor([target_idx], device=DEVICE)
        loss = -F.cross_entropy(logits, target_tensor)

        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = x_adv.grad.detach()

        # --- Smooth the gradient to get low-frequency noise ---
        grad_smooth = smooth_tensor(grad, kernel_size=LOWFREQ_SMOOTH)

        # PGD update
        x_adv = x_adv + ALPHA * torch.sign(grad_smooth)

        # project into epsilon ball
        perturb = torch.clamp(x_adv - x_orig, -EPSILON, EPSILON)
        x_adv = torch.clamp(x_orig + perturb, 0, 1).detach()

    return x_adv


# ------------------------------------------------------------
# Save image
# ------------------------------------------------------------
def save_img(tensor, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = transforms.ToPILImage()(tensor[0].cpu().clamp(0,1))
    img.save(out_path)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    rows = []

    images = []
    for class_dir in sorted(DATA_DIR.iterdir()):
        if class_dir.is_dir():
            for patt in ["*.JPEG", "*.jpg", "*.jpeg", "*.JPG"]:
                images.extend(class_dir.glob(patt))

    print(f"Found {len(images)} images.")

    if len(images) == 0:
        print("ERROR: No images found.")
        return

    # ---------------------------------------
    # Iterate through models
    # ---------------------------------------
    for model_name in MODEL_NAMES:
        print("\n==============================")
        print("MODEL:", model_name)
        print("==============================")

        try:
            model = timm.create_model(model_name, pretrained=True).eval().to(DEVICE)
        except Exception as e:
            print("Model failed:", e)
            continue

        preprocess, mean_list, std_list = build_preprocess(model)

        mean = torch.tensor(mean_list).view(1,3,1,1).to(DEVICE)
        std = torch.tensor(std_list).view(1,3,1,1).to(DEVICE)

        out_model_dir = OUT_DIR / model_name.replace("/", "_")

        for imgpath in images:
            img = Image.open(imgpath).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(DEVICE)

            wnid = imgpath.parent.name
            true_label = IMAGEWOOF_LABELS[wnid]

            with torch.no_grad():
                logits = model((x - mean) / std)
                clean_idx = logits.argmax(1).item()
                clean_name = decode_pred(clean_idx)

            clean_correct = clean_name == true_label

            target_idx, target_name = pick_target_class(model, x, mean, std, true_label)
            if target_idx is None:
                continue

            x_adv = targeted_pgd_smooth(model, x, mean, std, target_idx)

            with torch.no_grad():
                logits_adv = model((x_adv - mean) / std)
                adv_idx = logits_adv.argmax(1).item()
                adv_name = decode_pred(adv_idx)

            adv_success = adv_name == target_name

            # Save images
            save_img(x, out_model_dir / "clean" / wnid / imgpath.name)
            save_img(x_adv, out_model_dir / "adv" / wnid / imgpath.name)

            rows.append({
                "model": model_name,
                "image": imgpath.name,
                "wnid": wnid,
                "true_label": true_label,
                "clean_pred": clean_name,
                "clean_correct": clean_correct,
                "target_label": target_name,
                "adv_pred": adv_name,
                "target_success": adv_success
            })

    pd.DataFrame(rows).to_csv(OUT_DIR / "results.csv", index=False)
    print("\nSaved:", OUT_DIR / "results.csv")


if __name__ == "__main__":
    main()