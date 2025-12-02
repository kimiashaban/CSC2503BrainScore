import os
import random
from pathlib import Path
import shutil

# -----------------------------
# CONFIG
# -----------------------------
SRC = Path("imagewoof2-320/val")      # your local ImageWoof val folder
DEST = Path("imagewoof_sample_100")   # output folder
IMAGES_PER_CLASS = 10                 # 10 breeds × 10 images = 100 total

random.seed(42)  # for reproducibility


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"Source folder not found: {SRC}")

    DEST.mkdir(exist_ok=True)

    total_selected = 0

    # loop over breed folders (synset folders)
    for breed_dir in sorted(SRC.iterdir()):
        if not breed_dir.is_dir():
            continue

        images = sorted(list(breed_dir.glob("*.JPEG")))
        if len(images) == 0:
            continue

        # pick 10 random images from this synset
        chosen = random.sample(images, IMAGES_PER_CLASS)

        # create output subfolder for this synset
        outdir = DEST / breed_dir.name
        outdir.mkdir(parents=True, exist_ok=True)

        # copy the selected images
        for img in chosen:
            shutil.copy(img, outdir / img.name)
            total_selected += 1

        print(f"Selected {IMAGES_PER_CLASS} images from {breed_dir.name}")

    print("\n======================================")
    print(f"Done! Total images selected: {total_selected}")
    print(f"Saved to: {DEST}")
    print("======================================\n")


if __name__ == "__main__":
    main()