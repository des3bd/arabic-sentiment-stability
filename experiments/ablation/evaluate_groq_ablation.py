from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASELINE_PATH = Path("experiments/results/predictions_groq_gpt_oss.csv")
FEWSHOT_PATH = Path("experiments/ablation/results/predictions_groq_fewshot.csv")
OUTPUT_DIR = Path("experiments/ablation/results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = ["positive", "negative", "neutral"]


def normalize_label(label):
    label = str(label).strip().lower()

    if label in VALID_LABELS:
        return label
    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"

    return "invalid"


def prepare_df(path, configuration):
    df = pd.read_csv(path)

    df["configuration"] = configuration
    df["gold_label"] = df["gold_label"].astype(str).str.strip().str.lower()
    df["original_prediction"] = df["original_prediction"].apply(normalize_label)
    df["paraphrase_prediction"] = df["paraphrase_prediction"].apply(normalize_label)

    df["is_valid_original"] = df["original_prediction"].isin(VALID_LABELS)
    df["is_valid_paraphrase"] = df["paraphrase_prediction"].isin(VALID_LABELS)
    df["is_valid_pair"] = df["is_valid_original"] & df["is_valid_paraphrase"]

    df["consistent"] = df["original_prediction"] == df["paraphrase_prediction"]
    df["flip"] = ~df["consistent"]

    return df


def compute_metrics(df):
    rows = []

    for config, group in df.groupby("configuration"):
        valid_original = group[group["is_valid_original"]]
        valid_paraphrase = group[group["is_valid_paraphrase"]]
        valid_pair = group[group["is_valid_pair"]]

        rows.append({
            "configuration": config,
            "n_total": len(group),
            "n_valid_pairs": len(valid_pair),
            "original_accuracy": round(
                accuracy_score(valid_original["gold_label"], valid_original["original_prediction"]),
                4
            ),
            "paraphrase_accuracy": round(
                accuracy_score(valid_paraphrase["gold_label"], valid_paraphrase["paraphrase_prediction"]),
                4
            ),
            "original_macro_f1": round(
                f1_score(
                    valid_original["gold_label"],
                    valid_original["original_prediction"],
                    labels=VALID_LABELS,
                    average="macro",
                    zero_division=0,
                ),
                4
            ),
            "paraphrase_macro_f1": round(
                f1_score(
                    valid_paraphrase["gold_label"],
                    valid_paraphrase["paraphrase_prediction"],
                    labels=VALID_LABELS,
                    average="macro",
                    zero_division=0,
                ),
                4
            ),
            "consistency_rate": round(valid_pair["consistent"].mean(), 4),
            "flip_rate": round(valid_pair["flip"].mean(), 4),
            "invalid_original": int((~group["is_valid_original"]).sum()),
            "invalid_paraphrase": int((~group["is_valid_paraphrase"]).sum()),
        })

    return pd.DataFrame(rows)


def compute_per_class_consistency(df):
    rows = []

    valid_df = df[df["is_valid_pair"]].copy()

    for (config, label), group in valid_df.groupby(["configuration", "gold_label"]):
        rows.append({
            "configuration": config,
            "gold_label": label,
            "n": len(group),
            "consistent_count": int(group["consistent"].sum()),
            "flip_count": int(group["flip"].sum()),
            "consistency_rate": round(group["consistent"].mean(), 4),
            "flip_rate": round(group["flip"].mean(), 4),
        })

    return pd.DataFrame(rows)


def compute_per_class_f1(df):
    rows = []

    for config, group in df.groupby("configuration"):
        valid_original = group[group["is_valid_original"]]
        valid_paraphrase = group[group["is_valid_paraphrase"]]

        original_report = classification_report(
            valid_original["gold_label"],
            valid_original["original_prediction"],
            labels=VALID_LABELS,
            output_dict=True,
            zero_division=0,
        )

        paraphrase_report = classification_report(
            valid_paraphrase["gold_label"],
            valid_paraphrase["paraphrase_prediction"],
            labels=VALID_LABELS,
            output_dict=True,
            zero_division=0,
        )

        for label in VALID_LABELS:
            rows.append({
                "configuration": config,
                "label": label,
                "original_f1": round(original_report[label]["f1-score"], 4),
                "paraphrase_f1": round(paraphrase_report[label]["f1-score"], 4),
            })

    return pd.DataFrame(rows)


def main():
    baseline = prepare_df(BASELINE_PATH, "zero-shot")
    fewshot = prepare_df(FEWSHOT_PATH, "few-shot")

    combined = pd.concat([baseline, fewshot], ignore_index=True)

    combined_path = OUTPUT_DIR / "groq_ablation_combined_predictions.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    metrics = compute_metrics(combined)
    metrics_path = OUTPUT_DIR / "groq_ablation_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    per_class_consistency = compute_per_class_consistency(combined)
    per_class_consistency_path = OUTPUT_DIR / "groq_ablation_per_class_consistency.csv"
    per_class_consistency.to_csv(per_class_consistency_path, index=False, encoding="utf-8-sig")

    per_class_f1 = compute_per_class_f1(combined)
    per_class_f1_path = OUTPUT_DIR / "groq_ablation_per_class_f1.csv"
    per_class_f1.to_csv(per_class_f1_path, index=False, encoding="utf-8-sig")

    print("Saved:", combined_path)
    print("Saved:", metrics_path)
    print("Saved:", per_class_consistency_path)
    print("Saved:", per_class_f1_path)

    print("\nAblation metrics:")
    print(metrics.to_string(index=False))

    print("\nPer-class consistency:")
    print(per_class_consistency.to_string(index=False))

    print("\nPer-class F1:")
    print(per_class_f1.to_string(index=False))


if __name__ == "__main__":
    main()