#!/usr/bin/env python3
"""
CakraWatch - Baseline TF-IDF hoax classifier (Bahasa Indonesia)
Kartika Cyber Hackathon 2026

Datasets (in ../../data/):
- github_jibranfawaid_scrapping.csv  (JibranFawaid/turnbackhoax-dataset, 2021)
  Label 0 = hoaks (klaim/narasi hoaks), Label 1 = valid (artikel klarifikasi/fakta).
  Verified by manual content inspection 2026-10-21 - see DATASET_NOTES.md.
- raihankalla_turnbackhoax_raw.csv (3,760 TurnBackHoax debunk articles 2024)
  HOAX-ONLY corpus. Titles carry verdict prefixes ([SALAH]/[PENIPUAN]/...) which are
  stripped before use (label leakage). Used only in the augmented run.

Runs:
  A) GitHub dataset only (Headline + Body)
  B) GitHub + raihankalla titles/previews as extra 'hoaks' examples

Model: TF-IDF (word n-grams 1-2) + Logistic Regression (class_weight balanced).
Split: stratified 80/20, random_state=42.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[2] / "data"


def load_github() -> pd.DataFrame:
    df = pd.read_csv(DATA / "github_jibranfawaid_scrapping.csv", sep=";",
                     engine="python", on_bad_lines="skip", encoding="latin-1")
    df = df[["Label", "Headline", "Body"]].dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)
    df["label"] = df["Label"].map({0: "hoaks", 1: "valid"})
    df["text"] = (df["Headline"].fillna("") + " " + df["Body"].fillna("")).str.strip()
    return df[["text", "label"]]


def load_raihankalla_hoax() -> pd.DataFrame:
    df = pd.read_csv(DATA / "raihankalla_turnbackhoax_raw.csv")
    # strip verdict prefix "[SALAH] ", "[PENIPUAN] ", ... -> label leakage
    title = df["TITLE"].fillna("").str.replace(r"^\s*\[[^\]]*\]\s*", "", regex=True)
    df["text"] = (title + " " + df["PREVIEW"].fillna("")).str.strip()
    df["label"] = "hoaks"
    return df[["text", "label"]]


def run(df: pd.DataFrame, name: str):
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"])
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                          sublinear_tf=True, min_df=2)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(vec.fit_transform(X_tr), y_tr)
    pred = clf.predict(vec.transform(X_te))
    print(f"\n===== RUN {name} =====")
    print(f"total={len(df)}  train={len(X_tr)}  test={len(X_te)}")
    print("class counts:", df["label"].value_counts().to_dict())
    print(f"accuracy: {accuracy_score(y_te, pred):.4f}")
    print(classification_report(y_te, pred, digits=4))
    return accuracy_score(y_te, pred)


def main():
    gh = load_github()
    run(gh, "A: GitHub JibranFawaid only")
    aug = pd.concat([gh, load_raihankalla_hoax()], ignore_index=True)
    run(aug, "B: GitHub + raihankalla hoax titles (augmented)")


if __name__ == "__main__":
    sys.exit(main())
