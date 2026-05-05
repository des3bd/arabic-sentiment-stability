from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed" / "combined_sentiment_dataset.csv"
AGG_METRICS = ROOT / "experiments" / "results" / "aggregate_metrics.csv"
COMBINED_PREDS = ROOT / "experiments" / "results" / "all_model_predictions_combined.csv"
ERROR_SUMMARY = ROOT / "analysis" / "results" / "error_category_summary.csv"
ABLATION_METRICS = ROOT / "experiments" / "ablation" / "results" / "groq_ablation_metrics.csv"
MCNEMAR = ROOT / "analysis" / "results" / "mcnemar_tests.csv"

OUT_ASSETS = ROOT / "presentation" / "assets"
OUT_TABLES = ROOT / "presentation" / "tables"

OUT_ASSETS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)


def pct(x):
    return f"{x * 100:.2f}\\%"


def save_latex_table(df, path, index=False):
    latex = df.to_latex(index=index, escape=False)
    path.write_text(latex, encoding="utf-8")


def make_dataset_overview():
    df = pd.read_csv(DATA_PROCESSED)

    total = len(df)
    astd = (df["dataset"] == "ASTD").sum()
    ars = (df["dataset"] == "ArSarcasm").sum()

    pos = (df["sentiment_unified"] == "positive").sum()
    neg = (df["sentiment_unified"] == "negative").sum()
    neu = (df["sentiment_unified"] == "neutral").sum()

    overview = pd.DataFrame(
        {
            "Item": [
                "Total tweets",
                "ASTD tweets",
                "ArSarcasm tweets",
                "Positive",
                "Negative",
                "Neutral",
                "Number of dialect values",
            ],
            "Value": [
                total,
                astd,
                ars,
                pos,
                neg,
                neu,
                df["dialect"].nunique(),
            ],
        }
    )

    save_latex_table(overview, OUT_TABLES / "dataset_overview.tex", index=False)


def make_aggregate_metrics_table():
    df = pd.read_csv(AGG_METRICS).copy()

    keep = [
        "model",
        "original_accuracy",
        "paraphrase_accuracy",
        "original_macro_f1",
        "paraphrase_macro_f1",
        "consistency_rate",
        "flip_rate",
    ]
    df = df[keep]

    rename = {
        "model": "Model",
        "original_accuracy": "Orig. Acc.",
        "paraphrase_accuracy": "Para. Acc.",
        "original_macro_f1": "Orig. Macro-F1",
        "paraphrase_macro_f1": "Para. Macro-F1",
        "consistency_rate": "Consistency",
        "flip_rate": "Flip Rate",
    }
    df = df.rename(columns=rename)

    for col in [
        "Orig. Acc.",
        "Para. Acc.",
        "Orig. Macro-F1",
        "Para. Macro-F1",
        "Consistency",
        "Flip Rate",
    ]:
        df[col] = df[col].apply(pct)

    save_latex_table(df, OUT_TABLES / "aggregate_metrics.tex", index=False)


def make_error_summary_table():
    df = pd.read_csv(ERROR_SUMMARY).copy()

    if "percentage" in df.columns:
        df["percentage"] = df["percentage"].apply(lambda x: f"{x:.2f}\\%")
    elif "percentage_within_all" in df.columns:
        df["percentage_within_all"] = df["percentage_within_all"].apply(lambda x: f"{x:.2f}\\%")

    df.columns = [c.replace("_", " ").title() for c in df.columns]
    save_latex_table(df, OUT_TABLES / "error_category_summary.tex", index=False)


def make_ablation_table():
    df = pd.read_csv(ABLATION_METRICS).copy()

    keep = [
        "configuration",
        "original_accuracy",
        "paraphrase_accuracy",
        "original_macro_f1",
        "paraphrase_macro_f1",
        "consistency_rate",
        "flip_rate",
    ]
    df = df[keep]

    rename = {
        "configuration": "Configuration",
        "original_accuracy": "Orig. Acc.",
        "paraphrase_accuracy": "Para. Acc.",
        "original_macro_f1": "Orig. Macro-F1",
        "paraphrase_macro_f1": "Para. Macro-F1",
        "consistency_rate": "Consistency",
        "flip_rate": "Flip Rate",
    }
    df = df.rename(columns=rename)

    for col in [
        "Orig. Acc.",
        "Para. Acc.",
        "Orig. Macro-F1",
        "Para. Macro-F1",
        "Consistency",
        "Flip Rate",
    ]:
        df[col] = df[col].apply(pct)

    save_latex_table(df, OUT_TABLES / "ablation_metrics.tex", index=False)


def make_mcnemar_table():
    df = pd.read_csv(MCNEMAR).copy()

    out = pd.DataFrame(
        {
            "Comparison": df["comparison"],
            "p-value": df["p_value"].apply(lambda x: f"{x:.4f}"),
            "Significant?": df["significant_at_0.05"].apply(lambda x: "Yes" if x else "No"),
        }
    )

    save_latex_table(out, OUT_TABLES / "mcnemar_results.tex", index=False)


def plot_consistency_by_model():
    df = pd.read_csv(AGG_METRICS).copy()

    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["consistency_rate"] * 100)
    plt.ylabel("Consistency Rate (%)")
    plt.title("Consistency Rate by Model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_ASSETS / "consistency_by_model.png", dpi=300)
    plt.close()


def plot_accuracy_vs_consistency():
    df = pd.read_csv(AGG_METRICS).copy()

    x = np.arange(len(df))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, df["original_accuracy"] * 100, width, label="Original Accuracy")
    plt.bar(x, df["paraphrase_accuracy"] * 100, width, label="Paraphrase Accuracy")
    plt.bar(x + width, df["consistency_rate"] * 100, width, label="Consistency")

    plt.xticks(x, df["model"], rotation=15, ha="right")
    plt.ylabel("Score (%)")
    plt.title("Accuracy and Consistency by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ASSETS / "accuracy_vs_consistency.png", dpi=300)
    plt.close()


def plot_per_class_consistency():
    df = pd.read_csv(COMBINED_PREDS).copy()

    # make sure columns exist
    needed = {"model", "gold_label", "original_prediction", "paraphrase_prediction"}
    if not needed.issubset(df.columns):
        # try fallback if column name differs
        if "model_name" in df.columns:
            df = df.rename(columns={"model_name": "model"})
        if not needed.issubset(df.columns):
            raise ValueError("Combined predictions file does not contain the required columns.")

    df["consistent"] = df["original_prediction"] == df["paraphrase_prediction"]

    grouped = (
        df.groupby(["model", "gold_label"])["consistent"]
        .mean()
        .reset_index()
    )

    labels_order = ["negative", "neutral", "positive"]
    models = grouped["model"].unique()
    x = np.arange(len(labels_order))
    width = 0.25

    plt.figure(figsize=(9, 5))

    for i, model in enumerate(models):
        sub = grouped[grouped["model"] == model].copy()
        sub["gold_label"] = pd.Categorical(sub["gold_label"], categories=labels_order, ordered=True)
        sub = sub.sort_values("gold_label")
        plt.bar(x + (i - 1) * width, sub["consistent"] * 100, width, label=model)

    plt.xticks(x, labels_order)
    plt.ylabel("Consistency Rate (%)")
    plt.title("Per-Class Consistency by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ASSETS / "per_class_consistency.png", dpi=300)
    plt.close()


def plot_error_categories():
    df = pd.read_csv(ERROR_SUMMARY).copy()

    count_col = "count"
    category_col = "error_category"

    plt.figure(figsize=(9, 5))
    plt.bar(df[category_col], df[count_col])
    plt.ylabel("Count")
    plt.title("Expanded Error Categories")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_ASSETS / "error_categories.png", dpi=300)
    plt.close()


def plot_ablation_consistency():
    df = pd.read_csv(ABLATION_METRICS).copy()

    plt.figure(figsize=(7, 5))
    x = np.arange(len(df))
    width = 0.35

    plt.bar(x - width / 2, df["consistency_rate"] * 100, width, label="Consistency")
    plt.bar(x + width / 2, df["flip_rate"] * 100, width, label="Flip Rate")

    plt.xticks(x, df["configuration"])
    plt.ylabel("Rate (%)")
    plt.title("GPT-OSS Ablation: Zero-shot vs Few-shot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ASSETS / "ablation_consistency.png", dpi=300)
    plt.close()


def main():
    make_dataset_overview()
    make_aggregate_metrics_table()
    make_error_summary_table()
    make_ablation_table()
    make_mcnemar_table()

    plot_consistency_by_model()
    plot_accuracy_vs_consistency()
    plot_per_class_consistency()
    plot_error_categories()
    plot_ablation_consistency()

    print("Presentation assets generated successfully.")
    print(f"Tables saved to: {OUT_TABLES}")
    print(f"Figures saved to: {OUT_ASSETS}")


if __name__ == "__main__":
    main()