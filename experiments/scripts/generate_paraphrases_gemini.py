from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
from google import genai

SAMPLE_PATH = Path("experiments/data/sample_150.csv")
PROMPT_PATH = Path("experiments/prompts/paraphrase_gpt.txt")
OUTPUT_PATH = Path("experiments/results/paraphrases_gemini.csv")

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.7
RANDOM_SEED = 42


def load_prompt_template():
    return PROMPT_PATH.read_text(encoding="utf-8")


def clean_response(text):
    if text is None:
        return ""

    text = str(text).strip()

    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    return text


def call_gemini(client, prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": TEMPERATURE,
        },
    )

    return clean_response(response.text)


def main():
    client = genai.Client()

    df = pd.read_csv(SAMPLE_PATH)
    prompt_template = load_prompt_template()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        done_ids = set(existing["id"].astype(str))
        rows = existing.to_dict("records")
        print(f"Resuming. Existing paraphrases: {len(done_ids)}")
    else:
        done_ids = set()
        rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row["id"])

        if sample_id in done_ids:
            continue

        prompt = prompt_template.format(text=row["text"])

        try:
            paraphrase = call_gemini(client, prompt)
        except Exception as e:
            paraphrase = ""
            print(f"Error for {sample_id}: {e}")

        output_row = row.to_dict()
        output_row["paraphrase"] = paraphrase
        output_row["paraphrase_model"] = MODEL_NAME
        output_row["paraphrase_temperature"] = TEMPERATURE
        output_row["random_seed"] = RANDOM_SEED

        rows.append(output_row)

        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

        time.sleep(0.3)

    print(f"Saved paraphrases to: {OUTPUT_PATH}")
    print("Rows:", len(rows))


if __name__ == "__main__":
    main()