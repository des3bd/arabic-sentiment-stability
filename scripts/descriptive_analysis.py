from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/combined_sentiment_dataset.csv")
TABLES_DIR = Path("reports/tables")
FIGURES_DIR = Path("reports/figures")

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df, filename):
    output_path = TABLES_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved table: {output_path}")


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("BASIC INFO")
    print("=" * 60)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # -----------------------------
    # 1. Dataset summary
    # -----------------------------
    dataset_summary = (
        df.groupby("dataset")
        .agg(
            rows=("id", "count"),
            unique_texts=("text", "nunique"),
            sentiment_classes=("sentiment_unified", "nunique"),
            dialect_classes=("dialect", "nunique"),
        )
        .reset_index()
    )

    print("\nDataset summary:")
    print(dataset_summary)
    save_table(dataset_summary, "dataset_summary.csv")

    # -----------------------------
    # 2. Sentiment distribution
    # -----------------------------
    sentiment_distribution = (
        df["sentiment_unified"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    sentiment_distribution["percentage"] = (
        sentiment_distribution["count"] / len(df) * 100
    ).round(2)

    print("\nSentiment distribution:")
    print(sentiment_distribution)
    save_table(sentiment_distribution, "sentiment_distribution.csv")

    # -----------------------------
    # 3. Dataset by sentiment
    # -----------------------------
    dataset_sentiment = (
        df.groupby(["dataset", "sentiment_unified"])
        .size()
        .reset_index(name="count")
    )

    dataset_sentiment["percentage_within_dataset"] = (
        dataset_sentiment["count"]
        / dataset_sentiment.groupby("dataset")["count"].transform("sum")
        * 100
    ).round(2)

    print("\nDataset by sentiment:")
    print(dataset_sentiment)
    save_table(dataset_sentiment, "dataset_by_sentiment.csv")

    # -----------------------------
    # 4. Dialect distribution
    # -----------------------------
    dialect_distribution = (
        df["dialect"]
        .value_counts()
        .rename_axis("dialect")
        .reset_index(name="count")
    )

    dialect_distribution["percentage"] = (
        dialect_distribution["count"] / len(df) * 100
    ).round(2)

    print("\nDialect distribution:")
    print(dialect_distribution)
    save_table(dialect_distribution, "dialect_distribution.csv")

    # -----------------------------
    # 5. Dialect distribution only for ArSarcasm
    # -----------------------------
    arsarcasm_df = df[df["dataset"] == "ArSarcasm"]

    arsarcasm_dialect_distribution = (
        arsarcasm_df["dialect"]
        .value_counts()
        .rename_axis("dialect")
        .reset_index(name="count")
    )

    arsarcasm_dialect_distribution["percentage"] = (
        arsarcasm_dialect_distribution["count"] / len(arsarcasm_df) * 100
    ).round(2)

    print("\nArSarcasm dialect distribution:")
    print(arsarcasm_dialect_distribution)
    save_table(
        arsarcasm_dialect_distribution,
        "arsarcasm_dialect_distribution.csv"
    )

    # -----------------------------
    # 6. Sarcasm distribution
    # -----------------------------
    sarcasm_distribution = (
        df["sarcasm"]
        .value_counts(dropna=False)
        .rename_axis("sarcasm")
        .reset_index(name="count")
    )

    sarcasm_distribution["percentage"] = (
        sarcasm_distribution["count"] / len(df) * 100
    ).round(2)

    print("\nSarcasm distribution:")
    print(sarcasm_distribution)
    save_table(sarcasm_distribution, "sarcasm_distribution.csv")

    # -----------------------------
    # 7. Chart: sentiment distribution
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(
        sentiment_distribution["sentiment"],
        sentiment_distribution["count"]
    )
    plt.title("Unified Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Tweets")
    plt.tight_layout()

    sentiment_chart_path = FIGURES_DIR / "sentiment_distribution.png"
    plt.savefig(sentiment_chart_path, dpi=300)
    plt.close()

    print(f"Saved figure: {sentiment_chart_path}")

    # -----------------------------
    # 8. Chart: ArSarcasm dialect distribution
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(
        arsarcasm_dialect_distribution["dialect"],
        arsarcasm_dialect_distribution["count"]
    )
    plt.title("Dialect Distribution in ArSarcasm")
    plt.xlabel("Dialect")
    plt.ylabel("Number of Tweets")
    plt.tight_layout()

    dialect_chart_path = FIGURES_DIR / "arsarcasm_dialect_distribution.png"
    plt.savefig(dialect_chart_path, dpi=300)
    plt.close()

    print(f"Saved figure: {dialect_chart_path}")


if __name__ == "__main__":
    main()