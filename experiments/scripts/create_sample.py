from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/combined_sentiment_dataset.csv")
OUTPUT_DIR = Path("experiments/data")
OUTPUT_PATH = OUTPUT_DIR / "sample_150.csv"

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "id",
        "text",
        "sentiment_original",
        "sentiment_unified",
        "dataset",
        "dialect",
        "sarcasm",
        "source",
        "split",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    sampled_parts = []

    for label in ["positive", "negative", "neutral"]:
        class_df = df[df["sentiment_unified"] == label]

        if len(class_df) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Not enough rows for label {label}. "
                f"Available: {len(class_df)}, required: {SAMPLES_PER_CLASS}"
            )

        sampled = class_df.sample(
            n=SAMPLES_PER_CLASS,
            random_state=RANDOM_SEED
        )

        sampled_parts.append(sampled)

    sample_df = pd.concat(sampled_parts, ignore_index=True)

    sample_df = sample_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    sample_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved sample to: {OUTPUT_PATH}")
    print("Shape:", sample_df.shape)

    print("\nSentiment distribution:")
    print(sample_df["sentiment_unified"].value_counts())

    print("\nDataset distribution:")
    print(sample_df["dataset"].value_counts())

    print("\nDialect distribution:")
    print(sample_df["dialect"].value_counts())

    print("\nSarcasm distribution:")
    print(sample_df["sarcasm"].value_counts())


if __name__ == "__main__":
    main()