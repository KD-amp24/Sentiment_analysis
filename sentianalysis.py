"""
Phishing vs Safe Email Sentiment + Language Pattern Starter Pipeline (NLTK)

What this script does (Parts 1–3):
------------------------------------------------------------
Part 1 - Load & inspect:
  - Load CSV with pandas
  - Show first 5 rows
  - Show class counts (phishing vs safe/other)
  - Print 3 example phishing emails and 3 example safe emails

Part 2 - Preprocessing with NLTK:
  - Tokenize
  - Lowercase
  - Remove stopwords
  - Remove punctuation
  - Show original email + cleaned token list (sample)

Part 3 - Sentiment analysis with VADER:
  - Compute sentiment score (compound) for each email
  - Add sentiment_score column
  - Compute average sentiment by class (phishing vs safe)

How to run:
------------------------------------------------------------
python phishing_sentiment_pipeline.py --csv_path "your_file.csv"

Optional:
python phishing_sentiment_pipeline.py --csv_path "your_file.csv" --phishing_label "phishing" --safe_label "safe"
"""

import argparse
import re
import string
from typing import List, Optional

import pandas as pd

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer


# -------------------------------
# NLTK setup helpers
# -------------------------------
def ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources are available.
    If not present, download them.
    """
    # Punkt tokenizer data for word_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # In newer NLTK versions, punkt_tab may be needed depending on environment
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        # If your environment doesn't need it, this download is harmless.
        nltk.download("punkt_tab")

    # Stopwords corpus
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    # VADER lexicon for sentiment analysis
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon")


# -------------------------------
# Text preprocessing
# -------------------------------
def preprocess_email(
    text: str,
    stop_words: set,
) -> List[str]:
    """
    Preprocess a single email using the exact steps requested:
      1) Tokenization (NLTK word_tokenize)
      2) Lowercasing
      3) Remove stopwords
      4) Remove punctuation

    Returns:
      A list of cleaned tokens.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # 1) Tokenization
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for tok in tokens:
        # 2) Lowercasing
        tok = tok.lower()

        # 4) Remove punctuation (strict): remove tokens that are purely punctuation
        # Also strip punctuation off token edges (e.g., "urgent:" -> "urgent")
        tok = tok.strip(string.punctuation)

        # If token becomes empty after stripping punctuation, skip it
        if not tok:
            continue

        # 3) Remove stopwords
        if tok in stop_words:
            continue

        cleaned_tokens.append(tok)

    return cleaned_tokens


# -------------------------------
# Label normalization (optional)
# -------------------------------
def normalize_label(label_val: str) -> str:
    """
    Normalize labels into a consistent lowercase string.
    This helps if your labels are 'Phishing', 'PHISHING', 1, 0, etc.
    """
    if label_val is None:
        return ""
    return str(label_val).strip().lower()


def pick_safe_label(unique_labels: List[str], phishing_label: str, safe_label_arg: Optional[str]) -> Optional[str]:
    """
    Decide which label should be considered "safe" if not explicitly provided.

    Logic:
      - If user provided --safe_label, use it.
      - Otherwise, pick the first label that is not phishing_label.
      - If none found (e.g., dataset only contains phishing), return None.
    """
    if safe_label_arg:
        return safe_label_arg.lower()

    for lab in unique_labels:
        if lab != phishing_label:
            return lab
    return None


# -------------------------------
# Main pipeline
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="NLTK-based sentiment + language pattern inspection for phishing vs safe emails.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file containing columns: label, text")
    parser.add_argument("--phishing_label", type=str, default="phishing", help="Label value that represents phishing emails.")
    parser.add_argument("--safe_label", type=str, default=None, help="Label value that represents safe emails (optional).")
    parser.add_argument("--text_col", type=str, default="text", help="Name of the text column in the CSV.")
    parser.add_argument("--label_col", type=str, default="label", help="Name of the label column in the CSV.")
    parser.add_argument("--examples_n", type=int, default=3, help="How many example emails to print per class.")
    parser.add_argument("--show_preprocess_n", type=int, default=3, help="How many emails to show with original + cleaned token list.")
    args = parser.parse_args()

    # Ensure NLTK resources exist before we start
    ensure_nltk_resources()

    # ---------------------------------
    # Part 1: Load & inspect data
    # ---------------------------------
    df = pd.read_csv(args.csv_path)

    # Basic sanity checks for required columns
    if args.label_col not in df.columns or args.text_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{args.label_col}' and '{args.text_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalize labels for consistent comparisons
    df[args.label_col] = df[args.label_col].apply(normalize_label)

    print("\n=== Part 1: Dataset Loaded ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df[[args.label_col, args.text_col]].head(5))

    # Count phishing vs safe/other
    phishing_label = args.phishing_label.strip().lower()
    unique_labels = sorted(df[args.label_col].dropna().unique().tolist())
    safe_label = pick_safe_label(unique_labels, phishing_label, args.safe_label)

    print("\nLabel values found:", unique_labels)

    # Print counts for each label
    print("\nCounts by label:")
    print(df[args.label_col].value_counts(dropna=False))

    # If we have a safe label, also print phishing vs safe counts explicitly
    if safe_label is not None:
        phishing_count = int((df[args.label_col] == phishing_label).sum())
        safe_count = int((df[args.label_col] == safe_label).sum())
        print(f"\nPhishing count ('{phishing_label}'): {phishing_count}")
        print(f"Safe count ('{safe_label}'): {safe_count}")
    else:
        print("\nWARNING: Could not infer a safe label (dataset may only contain phishing).")

    # Print example emails
    def print_examples(label_name: str, n: int):
        subset = df[df[args.label_col] == label_name]
        if subset.empty:
            print(f"\nNo examples found for label '{label_name}'.")
            return
        print(f"\n--- {n} example emails for label '{label_name}' ---")
        for i, txt in enumerate(subset[args.text_col].head(n).tolist(), start=1):
            print(f"\nExample {i}:\n{txt}")

    print("\n=== Examples ===")
    print_examples(phishing_label, args.examples_n)
    if safe_label is not None:
        print_examples(safe_label, args.examples_n)

    # ---------------------------------
    # Part 2: Preprocessing with NLTK
    # ---------------------------------
    print("\n=== Part 2: Preprocessing with NLTK ===")

    stop_words = set(stopwords.words("english"))

    # Apply preprocessing to entire dataset (store tokens for later analysis)
    df["clean_tokens"] = df[args.text_col].apply(lambda t: preprocess_email(t, stop_words))

    # Show a few original + cleaned token lists
    show_n = min(args.show_preprocess_n, len(df))
    print(f"\nShowing original email + cleaned token list for {show_n} samples:\n")

    for idx in range(show_n):
        original = df.iloc[idx][args.text_col]
        tokens = df.iloc[idx]["clean_tokens"]
        label_val = df.iloc[idx][args.label_col]
        print(f"Sample {idx+1} (label={label_val})")
        print("Original email:")
        print(original)
        print("Cleaned token list:")
        print(tokens)
        print("-" * 60)

    # ---------------------------------
    # Part 3: Sentiment Analysis (VADER)
    # ---------------------------------
    print("\n=== Part 3: Sentiment Analysis (VADER) ===")

    sia = SentimentIntensityAnalyzer()

    # VADER returns: neg, neu, pos, compound
    # We'll use 'compound' as the overall sentiment_score requested.
    def vader_compound_score(text: str) -> float:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        return float(sia.polarity_scores(text)["compound"])

    df["sentiment_score"] = df[args.text_col].apply(vader_compound_score)

    # Compute averages by class
    phishing_avg = df.loc[df[args.label_col] == phishing_label, "sentiment_score"].mean()

    print(f"\nAverage sentiment_score for phishing ('{phishing_label}'): {phishing_avg:.4f}")

    if safe_label is not None:
        safe_avg = df.loc[df[args.label_col] == safe_label, "sentiment_score"].mean()
        print(f"Average sentiment_score for safe ('{safe_label}'): {safe_avg:.4f}")
    else:
        print("Safe label not available; skipping safe average sentiment.")

    # Optional: show a quick grouped summary for *all* labels found
    print("\nAverage sentiment_score by label (all classes):")
    print(df.groupby(args.label_col)["sentiment_score"].mean().sort_values(ascending=False))

    print("\nDone.")


if __name__ == "__main__":
    main()