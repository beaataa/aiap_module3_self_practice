"""
cleans the raw student data and builds the engineered features.

this mirrors the data cleaning section of eda.ipynb. everything here is
row-level or uses fixed rules from config.yaml, so it is safe to run on the
whole dataset before splitting. anything that has to learn a value from the
data (imputing attendance_rate, encoding, scaling) belongs in the training
pipeline instead, so it can be fitted on the training split only.
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(config):
    return pd.read_csv(PROJECT_ROOT / config["data"]["raw"])


def drop_unused(df, columns):
    # errors="ignore" so a rerun on already cleaned data doesn't blow up
    return df.drop(columns=columns, errors="ignore")


def drop_missing_target(df, target):
    # no way to train or score on a student with no final_test
    return df.dropna(subset=[target])


def standardise_categories(df, cleaning):
    df = df.copy()

    # CCA is typed in two styles, and the blanks are the no-cca students
    df["CCA"] = df["CCA"].str.upper().fillna(cleaning["cca_missing_fill"])

    # tuition is Y/N in some rows and Yes/No in others
    df["tuition"] = df["tuition"].replace(cleaning["tuition_map"])

    return df


def fix_age(df, cleaning):
    df = df.copy()

    # negative ages are unusable, 5 and 6 are 15 and 16 missing the leading 1
    df = df[df["age"] > 0]
    df["age"] = df["age"].replace(cleaning["age_typo_map"])

    # anything still outside 15/16 would be a new problem worth knowing about
    unexpected = set(df["age"].unique()) - set(cleaning["valid_ages"])
    if unexpected:
        raise ValueError(f"unexpected ages after cleaning: {sorted(unexpected)}")

    return df


def deduplicate(df, id_column, cleaning):
    # pass 1: rows identical in every remaining column. done before bag_color is
    # dropped, so two rows only count as duplicates if they really match. the
    # three columns already dropped above merge no extra rows, checked
    df = df.drop_duplicates()

    # what survives still has repeated student ids, differing only in bag_color
    # and attendance_rate. bag_color is dropped here, between the two passes
    df = drop_unused(df, cleaning["drop_after_exact_dedup"])

    # pass 2: one row per student. blanks sort last, so where a student has two
    # rows the one with a real attendance_rate is the one kept
    df = df.sort_values("attendance_rate", na_position="last")
    df = df.drop_duplicates(subset=id_column, keep="first")

    return df.sort_index()


def add_features(df, features):
    df = df.copy()

    male, female = features["class_size_from"]
    df["class_size"] = df[male] + df[female]

    sleep_col, wake_col = features["sleep_hours_from"]
    sleep = pd.to_datetime(df[sleep_col], format="%H:%M")
    wake = pd.to_datetime(df[wake_col], format="%H:%M")
    df["sleep_hours"] = ((wake - sleep).dt.total_seconds() / 3600) % 24

    # the originals are replaced by the two features above
    return drop_unused(df, [male, female, sleep_col, wake_col])


def prepare_data(config=None):
    """returns the cleaned dataframe, student_id included."""
    config = config or load_config()
    cleaning = config["cleaning"]

    df = load_raw(config)
    df = drop_unused(df, cleaning["drop_upfront"])
    df = drop_missing_target(df, config["target"])
    df = standardise_categories(df, cleaning)
    df = fix_age(df, cleaning)
    df = deduplicate(df, config["id_column"], cleaning)
    df = add_features(df, config["features"])

    return df.reset_index(drop=True)


def main():
    config = load_config()
    df = prepare_data(config)

    out_path = PROJECT_ROOT / config["data"]["cleaned"]
    df.to_csv(out_path, index=False)

    print(f"rows: {len(df)}, columns: {len(df.columns)}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
