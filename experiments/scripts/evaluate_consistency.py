from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

RESULTS_DIR = Path("experiments/results")

PREDICTION_FILES = {
    "Gemini 2.5 Flash": RESULTS_DIR / "predictions_gemini.csv",
    "GPT-OSS-20B via Groq": RESULTS_DIR / "predictions_groq_gpt_oss.csv",
    "AraBERT-ArSAS": RESULTS_DIR / "predictions_arabert.csv",
}

VALID_LABELS = ["positive", "negative", "neutral"]


def normalize_prediction(label):
    label = str(label).strip().lower()

    if label in VALID_LABELS:
        return label

    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"

    # For AraBERT if mixed appears, keep it as invalid for main metrics
    return "invalid"


def load_predictions():
    all_frames = []

    for model_name, path in PREDICTION_FILES.items():
        if not path.exists():
            print(f"Missing file, skipping: {path}")
            continue

        df = pd.read_csv(path)

        df["model_name"] = model_name
        df["gold_label"] = df["gold_label"].astype(str).str.strip().str.lower()
        df["original_prediction"] = df["original_prediction"].apply(normalize_prediction)
        df["paraphrase_prediction"] = df["paraphrase_prediction"].apply(normalize_prediction)

        df["is_valid_original"] = df["original_prediction"].isin(VALID_LABELS)
        df["is_valid_paraphrase"] = df["paraphrase_prediction"].isin(VALID_LABELS)
        df["is_valid_pair"] = df["is_valid_original"] & df["is_valid_paraphrase"]

        df["consistent"] = df["original_prediction"] == df["paraphrase_prediction"]
        df["flip"] = ~df["consistent"]

        all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError("No prediction files found.")

    return pd.concat(all_frames, ignore_index=True)


def compute_aggregate_metrics(df):
    rows = []

    for model_name, model_df in df.groupby("model_name"):
        valid_original = model_df[model_df["is_valid_original"]]
        valid_paraphrase = model_df[model_df["is_valid_paraphrase"]]
        valid_pair = model_df[model_df["is_valid_pair"]]

        original_accuracy = accuracy_score(
            valid_original["gold_label"],
            valid_original["original_prediction"]
        )

        paraphrase_accuracy = accuracy_score(
            valid_paraphrase["gold_label"],
            valid_paraphrase["paraphrase_prediction"]
        )

        original_macro_f1 = f1_score(
            valid_original["gold_label"],
            valid_original["original_prediction"],
            labels=VALID_LABELS,
            average="macro",
            zero_division=0
        )

        paraphrase_macro_f1 = f1_score(
            valid_paraphrase["gold_label"],
            valid_paraphrase["paraphrase_prediction"],
            labels=VALID_LABELS,
            average="macro",
            zero_division=0
        )

        consistency_rate = valid_pair["consistent"].mean()
        flip_rate = valid_pair["flip"].mean()

        rows.append({
            "model": model_name,
            "n_total": len(model_df),
            "n_valid_pairs": len(valid_pair),
            "original_accuracy": round(original_accuracy, 4),
            "paraphrase_accuracy": round(paraphrase_accuracy, 4),
            "original_macro_f1": round(original_macro_f1, 4),
            "paraphrase_macro_f1": round(paraphrase_macro_f1, 4),
            "consistency_rate": round(consistency_rate, 4),
            "flip_rate": round(flip_rate, 4),
            "invalid_original": int((~model_df["is_valid_original"]).sum()),
            "invalid_paraphrase": int((~model_df["is_valid_paraphrase"]).sum()),
        })

    return pd.DataFrame(rows)


def compute_group_consistency(df, group_col, output_name):
    rows = []

    valid_df = df[df["is_valid_pair"]].copy()

    for (model_name, group_value), group_df in valid_df.groupby(["model_name", group_col]):
        rows.append({
            "model": model_name,
            group_col: group_value,
            "n": len(group_df),
            "consistent_count": int(group_df["consistent"].sum()),
            "flip_count": int(group_df["flip"].sum()),
            "consistency_rate": round(group_df["consistent"].mean(), 4),
            "flip_rate": round(group_df["flip"].mean(), 4),
        })

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / output_name, index=False, encoding="utf-8-sig")
    return result


def compute_per_class_f1(df):
    rows = []

    for model_name, model_df in df.groupby("model_name"):
        valid_original = model_df[model_df["is_valid_original"]]
        valid_paraphrase = model_df[model_df["is_valid_paraphrase"]]

        original_report = classification_report(
            valid_original["gold_label"],
            valid_original["original_prediction"],
            labels=VALID_LABELS,
            output_dict=True,
            zero_division=0
        )

        paraphrase_report = classification_report(
            valid_paraphrase["gold_label"],
            valid_paraphrase["paraphrase_prediction"],
            labels=VALID_LABELS,
            output_dict=True,
            zero_division=0
        )

        for label in VALID_LABELS:
            rows.append({
                "model": model_name,
                "label": label,
                "original_precision": round(original_report[label]["precision"], 4),
                "original_recall": round(original_report[label]["recall"], 4),
                "original_f1": round(original_report[label]["f1-score"], 4),
                "paraphrase_precision": round(paraphrase_report[label]["precision"], 4),
                "paraphrase_recall": round(paraphrase_report[label]["recall"], 4),
                "paraphrase_f1": round(paraphrase_report[label]["f1-score"], 4),
            })

    return pd.DataFrame(rows)


def extract_failure_cases(df):
    failures = df[
        df["is_valid_pair"]
        & (df["original_prediction"] != df["paraphrase_prediction"])
    ].copy()

    failures = failures[
        [
            "model_name",
            "id",
            "gold_label",
            "dataset",
            "dialect",
            "sarcasm",
            "original_text",
            "paraphrase",
            "original_prediction",
            "paraphrase_prediction",
        ]
    ]

    failures = failures.rename(columns={"model_name": "model"})

    return failures


def main():
    df = load_predictions()

    combined_path = RESULTS_DIR / "all_model_predictions_combined.csv"
    df.to_csv(combined_path, index=False, encoding="utf-8-sig")
    print(f"Saved combined predictions: {combined_path}")

    aggregate_metrics = compute_aggregate_metrics(df)
    aggregate_metrics_path = RESULTS_DIR / "aggregate_metrics.csv"
    aggregate_metrics.to_csv(aggregate_metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved aggregate metrics: {aggregate_metrics_path}")
    print("\nAggregate metrics:")
    print(aggregate_metrics)

    per_class_f1 = compute_per_class_f1(df)
    per_class_f1_path = RESULTS_DIR / "per_class_f1.csv"
    per_class_f1.to_csv(per_class_f1_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved per-class F1: {per_class_f1_path}")

    per_class_consistency = compute_group_consistency(
        df,
        "gold_label",
        "per_class_consistency.csv"
    )
    print("\nPer-class consistency:")
    print(per_class_consistency)

    per_dialect_consistency = compute_group_consistency(
        df,
        "dialect",
        "per_dialect_consistency.csv"
    )
    print("\nPer-dialect consistency:")
    print(per_dialect_consistency)

    per_sarcasm_consistency = compute_group_consistency(
        df,
        "sarcasm",
        "per_sarcasm_consistency.csv"
    )
    print("\nPer-sarcasm consistency:")
    print(per_sarcasm_consistency)

    failures = extract_failure_cases(df)
    failure_path = RESULTS_DIR / "failure_cases.csv"
    failures.to_csv(failure_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved failure cases: {failure_path}")
    print("Number of failure cases:", len(failures))

    print("\nFirst 10 failure cases:")
    print(failures.head(10))


if __name__ == "__main__":
    main()