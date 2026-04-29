# Arabic Sentiment Stability Under Paraphrasing

This project investigates whether Arabic sentiment analysis models maintain stable predictions when Arabic text is paraphrased. The main goal is to measure **sentiment drift**: cases where a model assigns one sentiment label to an original tweet, but a different label to its paraphrase.

The project uses Arabic Twitter data from ASTD and ArSarcasm, generates paraphrases, compares multiple sentiment models, performs error analysis, runs an ablation experiment, and applies statistical significance testing.

---

## Project Overview

Arabic sentiment analysis is challenging because Arabic social media includes:

- Modern Standard Arabic (MSA)
- Dialectal Arabic
- Informal spelling
- Sarcasm and irony
- Emojis
- Short and context-limited text
- News-like or factual tweets with emotionally loaded topics

Traditional metrics such as accuracy and F1-score show how well a model predicts labels, but they do not show whether the model is **robust** when the same meaning is expressed differently.

This project evaluates both:

1. **Classification performance**
   - Accuracy
   - Macro-F1

2. **Prediction stability**
   - Consistency rate
   - Flip rate

---

## Research Question

The main research question is:

> Do Arabic sentiment models maintain the same sentiment prediction when Arabic tweets are paraphrased?

Additional questions:

- Which models are most stable under paraphrasing?
- Which sentiment classes are most fragile?
- Do dialect, sarcasm, or informal wording affect stability?
- Can few-shot prompting improve robustness?

---

## Models Compared

The experiment compares three models:

| Model | Role |
|---|---|
| Gemini 2.5 Flash | General instruction-following LLM |
| GPT-OSS-20B via Groq | Open GPT-style LLM hosted through Groq |
| AraBERT-ArSAS | Arabic-specific transformer sentiment classifier |

Gemini 2.5 Flash was also used to generate one paraphrase per tweet.

---

## Dataset

The project combines two public Arabic sentiment datasets:

| Dataset | Description |
|---|---|
| ASTD | Arabic Sentiment Tweets Dataset |
| ArSarcasm | Arabic sarcasm and sentiment dataset with dialect and sarcasm metadata |

The processed combined dataset contains:

```text
20,241 tweets
```

Unified sentiment labels:

```text
positive
negative
neutral
```

The main processed dataset is stored at:

```text
data/processed/combined_sentiment_dataset.csv
```

More details are available in:

```text
data/README.md
```

---

## Experiment Sample

The baseline experiment uses a balanced sample of 150 tweets:

| Sentiment | Count |
|---|---:|
| Positive | 50 |
| Negative | 50 |
| Neutral | 50 |

Sample file:

```text
experiments/data/sample_150.csv
```

---

## Setup

Install the required Python packages:

```bash
pip install pandas scikit-learn tqdm transformers torch google-genai openai statsmodels
```

---

## API Keys

The project uses Gemini and Groq APIs.

Set the environment variables before running API scripts.

### Windows PowerShell

```powershell
setx GEMINI_API_KEY "your_gemini_key_here"
setx GROQ_API_KEY "your_groq_key_here"
```

Close and reopen PowerShell, then check:

```powershell
echo $env:GEMINI_API_KEY
echo $env:GROQ_API_KEY
```

Do not commit API keys to GitHub.

The `.gitignore` file should include:

```text
.env
*.env
data/raw/
__pycache__/
*.pyc
```

---

## Reproducing the Dataset

To prepare the combined dataset:

```bash
python scripts/prepare_dataset.py
```

To generate descriptive dataset statistics:

```bash
python scripts/descriptive_analysis.py
```

---

## Reproducing the Main Experiment

### 1. Create the sample

```bash
python experiments/scripts/create_sample.py
```

### 2. Generate paraphrases with Gemini

```bash
python experiments/scripts/generate_paraphrases_gemini.py
```

Output:

```text
experiments/results/paraphrases_gemini.csv
```

### 3. Run sentiment classification

Gemini:

```bash
python experiments/scripts/run_gemini_sentiment.py
```

GPT-OSS via Groq:

```bash
python experiments/scripts/run_groq_sentiment.py
```

AraBERT:

```bash
python experiments/scripts/run_arabert_sentiment.py
```

### 4. Evaluate consistency and classification metrics

```bash
python experiments/scripts/evaluate_consistency.py
```

This generates:

```text
experiments/results/aggregate_metrics.csv
experiments/results/per_class_f1.csv
experiments/results/per_class_consistency.csv
experiments/results/per_dialect_consistency.csv
experiments/results/per_sarcasm_consistency.csv
experiments/results/failure_cases.csv
```

---

## Baseline Results

The main baseline results are:

| Model | Original Accuracy | Paraphrase Accuracy | Original Macro-F1 | Paraphrase Macro-F1 | Consistency Rate | Flip Rate |
|---|---:|---:|---:|---:|---:|---:|
| AraBERT-ArSAS | 61.74% | 67.33% | 61.68% | 66.72% | 78.52% | 21.48% |
| GPT-OSS-20B via Groq | 73.33% | 73.33% | 72.96% | 72.76% | 85.33% | 14.67% |
| Gemini 2.5 Flash | 69.33% | 67.33% | 66.63% | 65.26% | 90.67% | 9.33% |

Key findings:

- Gemini had the highest consistency rate.
- GPT-OSS-20B had the strongest accuracy and macro-F1.
- AraBERT-ArSAS had the highest flip rate.
- Accuracy and stability were not always aligned.

---

## Expanded Error Analysis

The project identified 68 sentiment flip cases.

Main error categories:

| Error Category | Count | Percentage |
|---|---:|---:|
| Neutral-news confusion | 27 | 39.71% |
| Dialect/slang ambiguity | 17 | 25.00% |
| Sarcasm/irony ambiguity | 13 | 19.12% |
| Paraphrase clarified sentiment | 6 | 8.82% |
| Short text/context loss | 2 | 2.94% |
| Entity/topic bias | 1 | 1.47% |
| Paraphrase sentiment shift | 1 | 1.47% |
| Gold-label/context ambiguity | 1 | 1.47% |

To reproduce the expanded error analysis:

```bash
python analysis/scripts/categorize_errors.py
```

---

## Ablation Experiment

An ablation experiment tested whether few-shot prompting improves GPT-OSS stability.

Run the few-shot GPT-OSS experiment:

```bash
python experiments/ablation/run_groq_fewshot_sentiment.py
```

Evaluate the ablation:

```bash
python experiments/ablation/evaluate_groq_ablation.py
```

Ablation results:

| Configuration | Original Accuracy | Paraphrase Accuracy | Original Macro-F1 | Paraphrase Macro-F1 | Consistency Rate | Flip Rate |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 73.33% | 73.33% | 72.96% | 72.76% | 85.33% | 14.67% |
| Few-shot | 73.33% | 73.33% | 73.15% | 73.21% | 90.00% | 10.00% |

The few-shot prompt improved consistency but did not change accuracy.

---

## Statistical Significance Testing

McNemar's test was used to compare paired consistency outcomes.

Run:

```bash
python analysis/scripts/statistical_tests.py
```

Results:

| Comparison | p-value | Significant at 0.05? |
|---|---:|---|
| Gemini vs GPT-OSS zero-shot | 0.2005 | No |
| Gemini vs AraBERT | 0.0066 | Yes |
| GPT-OSS zero-shot vs few-shot | 0.1185 | No |

The difference between Gemini and AraBERT was statistically significant, while the other differences were not significant at the 0.05 level.

---

## Main Findings

The study found that:

1. Arabic sentiment predictions are not fully stable under paraphrasing.
2. Gemini was the most stable model.
3. GPT-OSS-20B achieved the best accuracy and macro-F1.
4. AraBERT improved after paraphrasing but had the highest flip rate.
5. Neutral-news confusion was the most common error category.
6. Dialect and slang ambiguity strongly affected prediction stability.
7. Few-shot prompting improved GPT-OSS consistency, especially for neutral tweets.
8. Some paraphrases clarified sentiment, while others weakened or shifted sentiment cues.

---

## Limitations

This project has several limitations:

- The experiment used a sample of 150 tweets.
- Paraphrases were generated using only one model.
- Some dialect categories had small sample sizes.
- Error categories were based on rule-assisted analysis and manual inspection.
- Some gold labels may be ambiguous.
- LLM performance may vary with different prompt wording.
- AraBERT-ArSAS has a `mixed` label, while the project evaluation uses only `positive`, `negative`, and `neutral`.

---

## Data Sources

- ASTD: https://github.com/mahmoudnabil/ASTD
- ArSarcasm: https://github.com/iabufarha/ArSarcasm

---

## Project Repository

GitHub:

```text
https://github.com/des3bd/arabic-sentiment-stability
```

---

## Citation

If referencing this project, cite the original datasets and related models used in the experiment:

- ASTD: Arabic Sentiment Tweets Dataset
- ArSarcasm: Arabic sarcasm and sentiment dataset
- Gemini API documentation
- Groq GPT-OSS documentation
- AraBERT-ArSAS Hugging Face model card