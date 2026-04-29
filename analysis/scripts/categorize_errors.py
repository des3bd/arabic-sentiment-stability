from pathlib import Path
import pandas as pd

INPUT_PATH = Path("experiments/results/failure_cases.csv")
OUTPUT_DIR = Path("analysis/results")
OUTPUT_CASES_PATH = OUTPUT_DIR / "error_cases_categorized.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "error_category_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def contains_any(text, keywords):
    text = str(text).lower()
    return any(keyword in text for keyword in keywords)


def categorize_error(row):
    original = str(row["original_text"])
    paraphrase = str(row["paraphrase"])
    combined = f"{original} {paraphrase}"

    original_pred = row["original_prediction"]
    paraphrase_pred = row["paraphrase_prediction"]
    gold = row["gold_label"]
    dialect = str(row["dialect"]).lower()
    sarcasm = str(row["sarcasm"]).lower()

    # 1. Sarcasm / irony / emoji
    sarcasm_keywords = [
        "😂", "🤣", "هههه", "مضحكه", "مسخرة", "خرافي", "عجيب",
        "يا سلام", "احلى", "شفوني"
    ]

    if sarcasm == "true" or contains_any(combined, sarcasm_keywords):
        return "sarcasm_irony_ambiguity"

    # 2. Neutral/news confusion
    news_keywords = [
        "خبر", "عاجل", "تقرير", "تفاصيل", "الرابط", "صورة",
        "يعتزم", "أعلن", "قال", "صرح", "مقتل", "استشهاد",
        "تحذير", "افتتاح", "تكريم", "جامعة", "مقال", "مشاهد"
    ]

    if gold == "neutral" or contains_any(combined, news_keywords):
        return "neutral_news_confusion"

    # 3. Dialect / slang ambiguity
    dialect_keywords = [
        "مش", "اوي", "ياجدعان", "يعني", "وين", "هني", "كيوت",
        "ناعم", "وش", "ما راح", "ماهو", "اتوقع", "بقولكم",
        "جمبكو", "جبت الجون", "ڤالنتينو"
    ]

    if dialect in ["egypt", "gulf", "levant", "magreb"] or contains_any(combined, dialect_keywords):
        return "dialect_slang_ambiguity"

    # 4. Short text / context loss
    original_len = len(original.split())
    paraphrase_len = len(paraphrase.split())

    if original_len <= 4 or paraphrase_len <= 4:
        return "short_text_or_context_loss"

    # 5. Paraphrase changed sentiment direction
    if original_pred != paraphrase_pred:
        if gold == original_pred:
            return "paraphrase_sentiment_shift"
        if gold == paraphrase_pred:
            return "paraphrase_clarified_sentiment"

    # 6. Positive/negative cues introduced
    positive_keywords = [
        "رائع", "مذهل", "جميل", "انتصار", "تألق", "الصادقة",
        "الأفضل", "استثنائية", "يبارك", "شكرا", "مبروك"
    ]

    negative_keywords = [
        "سيء", "جريمة", "مرفوض", "تحذير", "مقتل", "استشهاد",
        "بلطجية", "كارثة", "حريق", "نار", "سرطان"
    ]

    if contains_any(paraphrase, positive_keywords) or contains_any(paraphrase, negative_keywords):
        return "positive_negative_cue_added"

    # 7. Entity/topic bias
    entity_keywords = [
        "ترامب", "السيسي", "ميسي", "برشلونة", "ريال مدريد",
        "سوريا", "مصر", "إيران", "الأرجنتين", "القدس",
        "حلب", "غزة"
    ]

    if contains_any(combined, entity_keywords):
        return "entity_topic_bias"

    return "gold_label_or_context_ambiguity"


def add_short_explanation(category):
    explanations = {
        "paraphrase_sentiment_shift": "The paraphrase may have changed the sentiment strength or polarity.",
        "paraphrase_clarified_sentiment": "The paraphrase made the sentiment clearer than the original text.",
        "neutral_news_confusion": "A factual or news-like text was interpreted as emotional.",
        "sarcasm_irony_ambiguity": "Sarcasm, irony, emojis, or humor affected prediction stability.",
        "dialect_slang_ambiguity": "Dialectal or informal wording affected prediction stability.",
        "positive_negative_cue_added": "The paraphrase introduced stronger positive or negative cues.",
        "short_text_or_context_loss": "The text was too short or lacked enough context.",
        "entity_topic_bias": "Named entities or political/sports topics may have biased the prediction.",
        "gold_label_or_context_ambiguity": "The gold label or context appears ambiguous."
    }

    return explanations.get(category, "Uncategorized error.")


def main():
    df = pd.read_csv(INPUT_PATH)

    df["error_category"] = df.apply(categorize_error, axis=1)
    df["error_explanation"] = df["error_category"].apply(add_short_explanation)

    df.to_csv(OUTPUT_CASES_PATH, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["model", "error_category"])
        .size()
        .reset_index(name="count")
    )

    summary["percentage_within_model"] = (
        summary["count"]
        / summary.groupby("model")["count"].transform("sum")
        * 100
    ).round(2)

    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("Saved categorized cases to:", OUTPUT_CASES_PATH)
    print("Saved category summary to:", OUTPUT_SUMMARY_PATH)

    print("\nOverall category counts:")
    overall = (
        df["error_category"]
        .value_counts()
        .rename_axis("error_category")
        .reset_index(name="count")
    )
    overall["percentage"] = (overall["count"] / len(df) * 100).round(2)
    print(overall.to_string(index=False))

    print("\nCategory summary by model:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
