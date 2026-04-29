from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

print("=" * 60)
print("RAW FILES")
print("=" * 60)

for file in RAW_DIR.rglob("*"):
    if file.is_file():
        print(file)

print("\n" + "=" * 60)
print("ASTD SAMPLE")
print("=" * 60)

astd_path = RAW_DIR / "ASTD" / "data" / "Tweets.txt"

if astd_path.exists():
    astd = pd.read_csv(
        astd_path,
        sep="\t",
        header=None,
        names=["text", "sentiment_original"],
        encoding="utf-8",
        on_bad_lines="skip"
    )

    print(astd.head())
    print("\nASTD shape:", astd.shape)
    print("\nASTD labels:")
    print(astd["sentiment_original"].value_counts(dropna=False))
else:
    print("ASTD Tweets.txt not found:", astd_path)

print("\n" + "=" * 60)
print("ARSARCASM FILES PREVIEW")
print("=" * 60)

arsarcasm_dir = RAW_DIR / "ArSarcasm"

if arsarcasm_dir.exists():
    for file in arsarcasm_dir.rglob("*"):
        if file.suffix.lower() in [".csv", ".tsv", ".txt"]:
            print("\nFile:", file)
            try:
                if file.suffix.lower() == ".csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_csv(file, sep="\t")

                print(df.head())
                print("Shape:", df.shape)
                print("Columns:", list(df.columns))
            except Exception as e:
                print("Could not read file:", e)
else:
    print("ArSarcasm folder not found:", arsarcasm_dir)