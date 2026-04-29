from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

INPUT_PATH = Path("experiments/results/paraphrases_gemini.csv")
OUTPUT_PATH = Path("experiments/results/predictions_arabert.csv")

MODEL_NAME = "Abdo36/Arabert-Sentiment-Analysis-ArSAS"


def normalize_label(label):
    label = str(label).strip().lower()

    if "positive" in label or label == "pos":
        return "positive"
    if "negative" in label or label == "neg":
        return "negative"
    if "neutral" in label or label == "neu":
        return "neutral"

    # Some sentiment models may output mixed.
    # We keep it visible instead of forcing it into neutral.
    if "mixed" in label:
        return "mixed"

    return "invalid"


def classify_text(classifier, text):
    result = classifier(str(text), truncation=True)[0]
    raw_label = result["label"]
    score = result["score"]
    normalized = normalize_label(raw_label)
    return raw_label, normalized, score


def main():
    df = pd.read_csv(INPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_raw, original_pred, original_score = classify_text(
            classifier, row["text"]
        )

        paraphrase_raw, paraphrase_pred, paraphrase_score = classify_text(
            classifier, row["paraphrase"]
        )

        rows.append({
            "id": row["id"],
            "gold_label": row["sentiment_unified"],
            "dataset": row["dataset"],
            "dialect": row["dialect"],
            "sarcasm": row["sarcasm"],
            "original_text": row["text"],
            "paraphrase": row["paraphrase"],
            "model": MODEL_NAME,
            "temperature": "not_applicable",
            "original_prediction_raw": original_raw,
            "original_prediction": original_pred,
            "original_score": original_score,
            "paraphrase_prediction_raw": paraphrase_raw,
            "paraphrase_prediction": paraphrase_pred,
            "paraphrase_score": paraphrase_score,
        })

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved AraBERT predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()