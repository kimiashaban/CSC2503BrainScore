import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "results.csv"   # your combined CSV from user study
OUTPUT_DIR = Path("./analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

REQUIRED_COLS = [
    "participant",
    "trial_id",
    "model",
    "condition",
    "human_label",
    "true_label",
    "target_label",
    "image_path",
]

# Mapping: RAW MODEL NAME -> PRETTY NAME
MODEL_PRETTY = {
    "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384": "ConvNeXt-Large",
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnet50": "ResNet-50",
    "resnet18": "ResNet-18",
    "densenet121": "DenseNet-121",
    "none": "Clean Image"
}

# Brain-Scores per model
BRAIN_SCORES = {
    "ConvNeXt-Large": 0.47,
    "ConvNeXt-Tiny": 0.41,
    "ResNet-50": 0.39,
    "ResNet-18": 0.32,
    "DenseNet-121": 0.31,
}


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(csv_file):
    df = pd.read_csv(csv_file, header=None)

    if len(df.columns) == 8:
        df.columns = REQUIRED_COLS
    else:
        raise ValueError(
            f"CSV has {len(df.columns)} columns, expected 8.\nColumns: {df.columns}"
        )

    return df


# ============================================================
# 2. CLEAN + DERIVE FEATURES
# ============================================================

def preprocess(df):
    df["condition"] = df["condition"].str.strip().str.lower()

    df["correct"] = (df["human_label"] == df["true_label"]).astype(int)

    # Convert model → pretty name
    df["pretty_model"] = df["model"].map(MODEL_PRETTY)

    # Clean/adv flags
    df["is_clean"] = df["pretty_model"] == "Clean Image"
    df["is_adv"] = df["pretty_model"] != "Clean Image"

    return df


# ============================================================
# 3. METRICS
# ============================================================

def compute_metrics(df):

    # =========== Clean Accuracy ===========
    clean_df = df[df["is_clean"]]
    clean_accuracy = clean_df["correct"].mean()

    # =========== Adv per-model ===========
    adv_df = df[df["is_adv"]]

    adv_summary = (
        adv_df.groupby("pretty_model")["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "adv_accuracy"})
        .reset_index()
    )
    adv_summary["fooling_rate"] = 1 - adv_summary["adv_accuracy"]

    # Add Brain Score
    adv_summary["brain_score"] = adv_summary["pretty_model"].map(BRAIN_SCORES)

    # =========== Correlation ===========
    if adv_summary["brain_score"].notna().sum() >= 2:
        corr = adv_summary[["brain_score", "fooling_rate"]].corr().iloc[0, 1]
    else:
        corr = np.nan

    # =========== Per participant ===========
    per_participant = (
        df.groupby("participant")["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "overall_accuracy"})
    )

    # Clean vs Adv per participant
    pa_clean = (
        df[df["is_clean"]]
        .groupby("participant")["correct"]
        .mean()
        .rename("clean_accuracy")
    )
    pa_adv = (
        df[df["is_adv"]]
        .groupby("participant")["correct"]
        .mean()
        .rename("adv_accuracy")
    )
    participant_breakdown = pd.concat([pa_clean, pa_adv], axis=1)

    return {
        "clean_accuracy": clean_accuracy,
        "adv_summary": adv_summary,
        "corr": corr,
        "per_participant": per_participant,
        "participant_breakdown": participant_breakdown,
    }


# ============================================================
# 4. SAVE PLOTS + TABLES
# ============================================================

def save_outputs(summary):

    adv = summary["adv_summary"]

    # ------------------------------
    #  TABLE (LaTeX)
    # ------------------------------
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Human fooling rate per model.}")
    tex.append("\\begin{tabular}{lccc}")
    tex.append("\\toprule")
    tex.append("Model & Brain-Score & Adv. Accuracy & Fooling Rate \\\\")
    tex.append("\\midrule")

    for _, row in adv.iterrows():
        tex.append(
            f"{row['pretty_model']} & {row['brain_score']:.2f} & "
            f"{row['adv_accuracy']:.2f} & {row['fooling_rate']:.2f} \\\\"
        )

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    (OUTPUT_DIR / "table_results.tex").write_text("\n".join(tex))


    # ------------------------------
    #  PLOT — FOOLING RATES
    # ------------------------------
    x = adv["pretty_model"].tolist()
    y = adv["fooling_rate"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, color="cornflowerblue")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Fooling Rate")
    plt.title("Human Fooling Rate per Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fooling_rate.png", dpi=300)
    plt.close()


    # ------------------------------
    #  PLOT — BRAIN SCORE vs FOOLING
    # ------------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(adv["brain_score"], adv["fooling_rate"], s=120, color="crimson")

    for _, row in adv.iterrows():
        plt.text(
            row["brain_score"] + 0.005,
            row["fooling_rate"] + 0.005,
            row["pretty_model"],
            fontsize=9
        )

    plt.xlabel("Brain-Score")
    plt.ylabel("Fooling Rate")
    plt.title("Brain-Score vs Fooling Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brainscore_vs_fooling.png", dpi=300)
    plt.close()

    # Also save CSV
    adv.to_csv(OUTPUT_DIR / "adv_summary.csv", index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading CSV...")
    df = load_data(DATA_FILE)

    print("Preprocessing...")
    df = preprocess(df)

    print("Computing metrics...")
    summary = compute_metrics(df)

    print("\n=== CLEAN ACCURACY ===")
    print(summary["clean_accuracy"])

    print("\n=== ADVERSARIAL SUMMARY ===")
    print(summary["adv_summary"])

    print("\n=== PARTICIPANT BREAKDOWN ===")
    print(summary["participant_breakdown"])

    print("\nSaving outputs...")
    save_outputs(summary)

    print("\nDONE! All plots/tables saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()