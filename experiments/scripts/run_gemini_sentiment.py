from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
from google import genai

INPUT_PATH = Path("experiments/results/paraphrases_gemini.csv")
PROMPT_PATH = Path("experiments/prompts/sentiment_gemini_zero_shot.txt")
OUTPUT_PATH = Path("experiments/results/predictions_gemini.csv")

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.0


def load_prompt_template():
    return PROMPT_PATH.read_text(encoding="utf-8")


def normalize_label(label):
    label = str(label).strip().lower()

    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"

    return "invalid"


def call_gemini(client, prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": TEMPERATURE,
        },
    )

    return str(response.text).strip()


def classify_text(client, prompt_template, text):
    prompt = prompt_template.format(text=text)
    raw = call_gemini(client, prompt)
    return raw, normalize_label(raw)


def main():
    client = genai.Client()

    df = pd.read_csv(INPUT_PATH)
    prompt_template = load_prompt_template()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_raw, original_pred = classify_text(
            client, prompt_template, row["text"]
        )

        paraphrase_raw, paraphrase_pred = classify_text(
            client, prompt_template, row["paraphrase"]
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
            "temperature": TEMPERATURE,
            "original_prediction_raw": original_raw,
            "original_prediction": original_pred,
            "paraphrase_prediction_raw": paraphrase_raw,
            "paraphrase_prediction": paraphrase_pred,
        })

        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        time.sleep(0.2)

    print(f"Saved Gemini predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()