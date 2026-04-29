from pathlib import Path
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

RESULTS_DIR = Path("experiments/results")
ABLATION_DIR = Path("experiments/ablation/results")
OUTPUT_DIR = Path("analysis/results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_COMBINED_PATH = RESULTS_DIR / "all_model_predictions_combined.csv"
ABLATION_COMBINED_PATH = ABLATION_DIR / "groq_ablation_combined_predictions.csv"
OUTPUT_PATH = OUTPUT_DIR / "mcnemar_tests.csv"


def mcnemar_for_consistency(df_a, name_a, df_b, name_b):
    """
    McNemar test comparing two systems on whether each case was consistent.
    A case is consistent if original_prediction == paraphrase_prediction.
    """

    a = df_a[["id", "consistent"]].copy()
    b = df_b[["id", "consistent"]].copy()

    a = a.rename(columns={"consistent": f"{name_a}_consistent"})
    b = b.rename(columns={"consistent": f"{name_b}_consistent"})

    merged = a.merge(b, on="id", how="inner")

    a_col = f"{name_a}_consistent"
    b_col = f"{name_b}_consistent"

    both_correct = ((merged[a_col] == True) & (merged[b_col] == True)).sum()
    a_only = ((merged[a_col] == True) & (merged[b_col] == False)).sum()
    b_only = ((merged[a_col] == False) & (merged[b_col] == True)).sum()
    both_wrong = ((merged[a_col] == False) & (merged[b_col] == False)).sum()

    table = [[both_correct, a_only], [b_only, both_wrong]]

    # exact=True is safer for small discordant counts
    result = mcnemar(table, exact=True)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n": len(merged),
        f"{name_a}_consistent_only": int(a_only),
        f"{name_b}_consistent_only": int(b_only),
        "both_consistent": int(both_correct),
        "both_flipped": int(both_wrong),
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "significant_at_0.05": result.pvalue < 0.05,
    }


def main():
    baseline = pd.read_csv(BASELINE_COMBINED_PATH)

    gemini = baseline[baseline["model_name"] == "Gemini 2.5 Flash"].copy()
    groq_zero = baseline[baseline["model_name"] == "GPT-OSS-20B via Groq"].copy()
    arabert = baseline[baseline["model_name"] == "AraBERT-ArSAS"].copy()

    ablation = pd.read_csv(ABLATION_COMBINED_PATH)
    groq_few = ablation[ablation["configuration"] == "few-shot"].copy()
    groq_zero_ablation = ablation[ablation["configuration"] == "zero-shot"].copy()

    tests = []

    tests.append(
        mcnemar_for_consistency(
            gemini,
            "Gemini",
            groq_zero,
            "GPT_OSS_zero"
        )
    )

    tests.append(
        mcnemar_for_consistency(
            gemini,
            "Gemini",
            arabert,
            "AraBERT"
        )
    )

    tests.append(
        mcnemar_for_consistency(
            groq_zero_ablation,
            "GPT_OSS_zero",
            groq_few,
            "GPT_OSS_few"
        )
    )

    results = pd.DataFrame(tests)
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_PATH)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()