from pathlib import Path
import os
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

INPUT_PATH = Path("experiments/results/paraphrases_gemini.csv")
PROMPT_PATH = Path("experiments/prompts/sentiment_groq_few_shot.txt")
OUTPUT_PATH = Path("experiments/ablation/results/predictions_groq_fewshot.csv")

MODEL_NAME = "openai/gpt-oss-20b"
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


def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Set it using: "
            'setx GROQ_API_KEY "your_key_here"'
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def call_groq(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content.strip()


def classify_text(client, prompt_template, text):
    prompt = prompt_template.format(text=text)
    raw = call_groq(client, prompt)
    return raw, normalize_label(raw)


def main():
    client = get_client()

    df = pd.read_csv(INPUT_PATH)
    prompt_template = load_prompt_template()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["id"].astype(str))
        rows = existing.to_dict("records")
        print(f"Resuming. Existing predictions: {len(done_ids)}")
    else:
        done_ids = set()
        rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["id"])

        if sample_id in done_ids:
            continue

        try:
            original_raw, original_pred = classify_text(
                client, prompt_template, row["text"]
            )

            paraphrase_raw, paraphrase_pred = classify_text(
                client, prompt_template, row["paraphrase"]
            )

        except Exception as e:
            print(f"Error for {sample_id}: {e}")
            original_raw = ""
            original_pred = "error"
            paraphrase_raw = ""
            paraphrase_pred = "error"

        rows.append({
            "id": row["id"],
            "gold_label": row["sentiment_unified"],
            "dataset": row["dataset"],
            "dialect": row["dialect"],
            "sarcasm": row["sarcasm"],
            "original_text": row["text"],
            "paraphrase": row["paraphrase"],
            "model": MODEL_NAME,
            "configuration": "few-shot",
            "temperature": TEMPERATURE,
            "original_prediction_raw": original_raw,
            "original_prediction": original_pred,
            "paraphrase_prediction_raw": paraphrase_raw,
            "paraphrase_prediction": paraphrase_pred,
        })

        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

        time.sleep(0.3)

    print(f"Saved Groq few-shot predictions to: {OUTPUT_PATH}")
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()