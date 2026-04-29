# Dataset

This folder contains the dataset files and processing documentation for the project **Arabic Sentiment Stability Under Paraphrasing**.

The goal of this project is to study whether Arabic sentiment models maintain stable sentiment predictions when Arabic text is paraphrased. For Task 2, this folder focuses only on dataset preparation, documentation, and descriptive analysis. No generative paraphrasing or model experiments are included at this stage.

---

## 1. Data Sources

This project uses two publicly available Arabic sentiment datasets.

### 1.1 ASTD: Arabic Sentiment Tweets Dataset

ASTD is an Arabic Twitter sentiment dataset. In this project, we use the `Tweets.txt` file from the official ASTD GitHub repository.

Repository:

```text
https://github.com/mahmoudnabil/ASTD
```

The file contains Arabic tweet text and sentiment labels.

Original ASTD labels:

| Original Label | Meaning |
|---|---|
| `POS` | Positive |
| `NEG` | Negative |
| `NEUTRAL` | Neutral |
| `OBJ` | Objective |

In the downloaded version used for this project, ASTD contains **9,694 rows**.

---

### 1.2 ArSarcasm

ArSarcasm is an Arabic sarcasm and sentiment dataset. It contains Arabic tweets annotated with sentiment, sarcasm, dialect, source, and train/test split information.

Repository:

```text
https://github.com/iabufarha/ArSarcasm
```

Files used:

| File | Rows |
|---|---:|
| `ArSarcasm_train.csv` | 8,437 |
| `ArSarcasm_test.csv` | 2,110 |

Original ArSarcasm sentiment labels:

| Original Label | Meaning |
|---|---|
| `positive` | Positive |
| `negative` | Negative |
| `neutral` | Neutral |

In total, ArSarcasm contains **10,547 rows**.

---

## 2. Processed Dataset

The processed combined dataset is stored in:

```text
data/processed/combined_sentiment_dataset.csv
```

The combined dataset contains:

```text
20,241 rows
9 columns
```

The combined dataset was created by merging ASTD and ArSarcasm into one unified sentiment dataset.

---

## 3. Dataset Schema

| Column | Description |
|---|---|
| `id` | Unique row identifier generated during preprocessing |
| `text` | Arabic tweet text |
| `sentiment_original` | Sentiment label from the original dataset |
| `sentiment_unified` | Standardized sentiment label used in this project |
| `dataset` | Dataset source: `ASTD` or `ArSarcasm` |
| `dialect` | Dialect label when available; `unknown` for ASTD |
| `sarcasm` | Sarcasm label when available; `unknown` for ASTD |
| `source` | Original source field when available |
| `split` | Dataset split: `train`, `test`, or `full` |

---

## 4. Label Mapping

To make the two datasets compatible, the original sentiment labels were mapped into three unified classes: `positive`, `negative`, and `neutral`.

| Original Label | Dataset | Unified Label |
|---|---|---|
| `POS` | ASTD | `positive` |
| `NEG` | ASTD | `negative` |
| `NEUTRAL` | ASTD | `neutral` |
| `OBJ` | ASTD | `neutral` |
| `positive` | ArSarcasm | `positive` |
| `negative` | ArSarcasm | `negative` |
| `neutral` | ArSarcasm | `neutral` |

The ASTD label `OBJ` was mapped to `neutral` because objective tweets usually do not express a clear positive or negative sentiment. This mapping simplifies the label space and makes ASTD compatible with ArSarcasm.

---

## 5. Label Definitions

The unified sentiment labels are defined as follows.

| Label | Definition |
|---|---|
| `positive` | The text expresses praise, satisfaction, happiness, support, approval, or a favorable opinion. |
| `negative` | The text expresses criticism, anger, sadness, dissatisfaction, complaint, rejection, or an unfavorable opinion. |
| `neutral` | The text reports information, asks a question, describes an event, or does not clearly express positive or negative sentiment. |

---

## 6. Example Annotations

| Text | Unified Label | Reason |
|---|---|---|
| `أحب هذا المنتج جدًا` | `positive` | The sentence expresses clear approval and satisfaction. |
| `الخدمة سيئة جدًا` | `negative` | The sentence expresses dissatisfaction and criticism. |
| `تم افتتاح الفرع الجديد اليوم` | `neutral` | The sentence reports factual information without clear sentiment. |
| `ما هي مميزات درجة رجال الأعمال؟` | `neutral` | The sentence is a question and does not express a clear opinion. |

These examples are used to clarify the annotation guidelines for the unified sentiment labels.

---

## 7. Descriptive Statistics

### 7.1 Sentiment Distribution

| Sentiment | Count | Percentage |
|---|---:|---:|
| `neutral` | 12,615 | 62.33% |
| `negative` | 5,171 | 25.55% |
| `positive` | 2,455 | 12.13% |

The dataset is imbalanced, with neutral tweets forming the largest class.

---

### 7.2 Dataset Source Distribution

| Dataset | Count | Percentage |
|---|---:|---:|
| `ArSarcasm` | 10,547 | 52.11% |
| `ASTD` | 9,694 | 47.89% |

---

### 7.3 Dialect Distribution in the Combined Dataset

| Dialect | Count | Percentage |
|---|---:|---:|
| `unknown` | 9,694 | 47.89% |
| `msa` | 7,062 | 34.89% |
| `egypt` | 2,383 | 11.77% |
| `levant` | 551 | 2.72% |
| `gulf` | 519 | 2.56% |
| `magreb` | 32 | 0.16% |

The `unknown` dialect category comes from ASTD because ASTD does not provide dialect labels.

---

### 7.4 Dialect Distribution in ArSarcasm Only

| Dialect | Count | Percentage |
|---|---:|---:|
| `msa` | 7,062 | 66.96% |
| `egypt` | 2,383 | 22.59% |
| `levant` | 551 | 5.22% |
| `gulf` | 519 | 4.92% |
| `magreb` | 32 | 0.30% |

Dialect metadata is mainly available through ArSarcasm.

---

### 7.5 Sarcasm Distribution

| Sarcasm | Count | Percentage |
|---|---:|---:|
| `unknown` | 9,694 | 47.89% |
| `False` | 8,865 | 43.80% |
| `True` | 1,682 | 8.31% |

The `unknown` sarcasm category comes from ASTD because ASTD does not include sarcasm annotations.

---

## 8. Preprocessing Steps

The preprocessing script performs the following steps:

1. Loads ASTD from:

```text
data/raw/ASTD/data/Tweets.txt
```

2. Loads ArSarcasm from:

```text
data/raw/ArSarcasm/dataset/ArSarcasm_train.csv
data/raw/ArSarcasm/dataset/ArSarcasm_test.csv
```

3. Standardizes the column names.

4. Cleans tweet text by:
   - removing unnecessary surrounding quotation marks,
   - stripping leading and trailing spaces,
   - normalizing repeated whitespace.

5. Maps the original sentiment labels to unified sentiment labels.

6. Adds metadata fields:
   - `dataset`
   - `dialect`
   - `sarcasm`
   - `source`
   - `split`

7. Removes rows with missing text or missing unified sentiment labels.

8. Saves the final combined dataset to:

```text
data/processed/combined_sentiment_dataset.csv
```

---

## 9. Generated Files

The following processed dataset file is generated:

```text
data/processed/combined_sentiment_dataset.csv
```

The descriptive analysis script generates tables under:

```text
reports/tables/
```

Generated tables include:

| File | Description |
|---|---|
| `dataset_summary.csv` | Summary by dataset source |
| `sentiment_distribution.csv` | Distribution of unified sentiment labels |
| `dataset_by_sentiment.csv` | Sentiment distribution within each dataset |
| `dialect_distribution.csv` | Dialect distribution in the full combined dataset |
| `arsarcasm_dialect_distribution.csv` | Dialect distribution in ArSarcasm only |
| `sarcasm_distribution.csv` | Sarcasm distribution |

The descriptive analysis script also generates figures under:

```text
reports/figures/
```

Generated figures include:

| File | Description |
|---|---|
| `sentiment_distribution.png` | Bar chart of unified sentiment distribution |
| `arsarcasm_dialect_distribution.png` | Bar chart of ArSarcasm dialect distribution |

---

## 10. Reproducibility

To recreate the processed dataset, run the following command from the project root:

```bash
python scripts/prepare_dataset.py
```

To recreate the descriptive statistics tables and figures, run:

```bash
python scripts/descriptive_analysis.py
```

If your system uses `python3`, use:

```bash
python3 scripts/prepare_dataset.py
python3 scripts/descriptive_analysis.py
```

---

## 11. Annotation and Inter-Annotator Agreement

Both ASTD and ArSarcasm already provide sentiment annotations. Therefore, the original dataset labels are used as the main reference labels in this project.

---

## 12. Suitability for the Project

The combined dataset is suitable for the project because it provides Arabic text with sentiment labels, which is required for testing sentiment stability under paraphrasing.

ASTD contributes general Arabic sentiment tweets, while ArSarcasm adds useful metadata such as dialect and sarcasm. This supports the project’s interest in studying possible sentiment drift across different Arabic varieties and difficult cases such as sarcastic text.

The dataset is especially useful for the planned future stages of the project, where paraphrases will be generated and model predictions will be compared before and after paraphrasing.

---

## 13. Limitations and Potential Biases

Several limitations should be considered:

1. Both datasets are based on Twitter data, so the language is short, informal, and influenced by social media writing conventions.
2. The dataset is imbalanced, with neutral tweets forming the majority class.
3. ASTD does not provide dialect labels, so dialect-based analysis depends mainly on ArSarcasm.
4. Dialect distribution is imbalanced. MSA and Egyptian Arabic are much more frequent than Gulf, Levantine, and Maghrebi Arabic.
5. Sarcasm labels are only available in ArSarcasm, so sarcasm-related analysis cannot be applied to ASTD.
6. Mapping `OBJ` from ASTD to `neutral` simplifies the original ASTD label scheme and may hide differences between objective factual tweets and neutral subjective tweets.
7. Twitter data may include spelling variation, slang, hashtags, mentions, and informal writing, which may affect sentiment analysis models.
8. The dataset may reflect platform-specific and topic-specific biases from the time and sources of collection.

---