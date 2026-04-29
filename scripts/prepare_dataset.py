from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "combined_sentiment_dataset.csv"


def clean_text(text):
    """Basic text cleaning without changing meaning."""
    if pd.isna(text):
        return None

    text = str(text).strip()

    # Remove surrounding quotes if present
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    # Normalize extra spaces
    text = " ".join(text.split())

    return text


def load_astd():
    astd_path = RAW_DIR / "ASTD" / "data" / "Tweets.txt"

    astd = pd.read_csv(
        astd_path,
        sep="\t",
        header=None,
        names=["text", "sentiment_original"],
        encoding="utf-8",
        on_bad_lines="skip"
    )

    label_map = {
        "POS": "positive",
        "NEG": "negative",
        "NEUTRAL": "neutral",
        "OBJ": "neutral"
    }

    astd["text"] = astd["text"].apply(clean_text)
    astd["sentiment_original"] = astd["sentiment_original"].astype(str).str.strip()
    astd["sentiment_unified"] = astd["sentiment_original"].map(label_map)

    astd["dataset"] = "ASTD"
    astd["dialect"] = "unknown"
    astd["sarcasm"] = "unknown"
    astd["source"] = "ASTD"
    astd["split"] = "full"

    return astd[
        [
            "text",
            "sentiment_original",
            "sentiment_unified",
            "dataset",
            "dialect",
            "sarcasm",
            "source",
            "split"
        ]
    ]


def load_arsarcasm_file(path, split):
    df = pd.read_csv(path)

    df = df.rename(columns={"tweet": "text"})

    df["text"] = df["text"].apply(clean_text)
    df["sentiment_original"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["sentiment_unified"] = df["sentiment_original"]

    df["dataset"] = "ArSarcasm"
    df["split"] = split

    return df[
        [
            "text",
            "sentiment_original",
            "sentiment_unified",
            "dataset",
            "dialect",
            "sarcasm",
            "source",
            "split"
        ]
    ]


def load_arsarcasm():
    train_path = RAW_DIR / "ArSarcasm" / "dataset" / "ArSarcasm_train.csv"
    test_path = RAW_DIR / "ArSarcasm" / "dataset" / "ArSarcasm_test.csv"

    train = load_arsarcasm_file(train_path, "train")
    test = load_arsarcasm_file(test_path, "test")

    return pd.concat([train, test], ignore_index=True)


def main():
    astd = load_astd()
    arsarcasm = load_arsarcasm()

    combined = pd.concat([astd, arsarcasm], ignore_index=True)

    # Remove rows with missing text or missing unified label
    combined = combined.dropna(subset=["text", "sentiment_unified"])

    # Remove empty texts
    combined = combined[combined["text"].str.len() > 0]

    # Keep only the three main labels
    combined = combined[
        combined["sentiment_unified"].isin(["positive", "negative", "neutral"])
    ]

    # Create stable IDs
    combined = combined.reset_index(drop=True)
    combined.insert(0, "id", ["SENT_" + str(i + 1).zfill(6) for i in range(len(combined))])

    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_PATH)
    print("Shape:", combined.shape)

    print("\nSentiment distribution:")
    print(combined["sentiment_unified"].value_counts())

    print("\nDataset distribution:")
    print(combined["dataset"].value_counts())

    print("\nDialect distribution:")
    print(combined["dialect"].value_counts(dropna=False))

    print("\nColumns:")
    print(list(combined.columns))


if __name__ == "__main__":
    main()