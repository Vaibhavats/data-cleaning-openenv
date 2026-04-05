"""
Task 3 (Hard): Full Data Cleaning Pipeline
==========================================
A 100-row raw CRM export needs a complete cleaning pass:
  1. Duplicate rows (exact duplicates + near-duplicates by customer_id)
  2. Missing values in `age` and `signup_date`
  3. Age outliers (unrealistic values: < 10 or > 100)
  4. Inconsistent `city` casing ("new york", "NEW YORK", "New York" → "New York")
  5. Invalid emails (missing '@', wrong format)
  6. `purchase_amount` stored as strings with '$' prefix → must be numeric

Grader dimensions:
  - deduplication   (precision/recall on known duplicate rows)
  - missing_filled  (completeness of fills for age + signup_date)
  - age_validity    (no ages < 10 or > 100 remain)
  - city_standardisation (all city values in title-case canonical set)
  - email_validity  (invalid emails removed or flagged)
  - dtype_purchase  (`purchase_amount` is numeric)
"""
from __future__ import annotations
import re, random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


# ── Canonical city values ────────────────────────────────────────────────────
CANONICAL_CITIES = {"New York", "Los Angeles", "Chicago", "Houston", "Phoenix"}
CITY_VARIANTS: Dict[str, str] = {
    "new york": "New York", "NEW YORK": "New York", "newyork": "New York",
    "los angeles": "Los Angeles", "LOS ANGELES": "Los Angeles", "LA": "Los Angeles",
    "chicago": "Chicago", "CHICAGO": "Chicago",
    "houston": "Houston", "HOUSTON": "Houston",
    "phoenix": "Phoenix", "PHOENIX": "Phoenix",
}


def _valid_email(e: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(e)))


def _make_dataset() -> pd.DataFrame:
    np.random.seed(99)
    random.seed(99)
    n_base = 85  # unique real customers

    ages   = np.random.randint(20, 65, size=n_base).astype(float)
    cities_clean = np.random.choice(list(CANONICAL_CITIES), size=n_base)

    df = pd.DataFrame({
        "customer_id":    [f"U{3000+i}" for i in range(n_base)],
        "full_name":      [f"Name_{i}" for i in range(n_base)],
        "email":          [f"user{i}@example.com" for i in range(n_base)],
        "age":            ages,
        "city":           cities_clean,
        "purchase_amount": [f"${np.round(np.random.uniform(20, 800), 2)}" for _ in range(n_base)],
        "signup_date":    pd.date_range("2021-01-01", periods=n_base, freq="4D").strftime("%Y-%m-%d"),
    })

    # 1. Inject 7 duplicate rows (exact)
    dup_indices = [5, 12, 20, 33, 44, 60, 72]
    dup_rows    = df.iloc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)  # now 92 rows

    # 2. Add 8 more near-duplicate rows (same customer_id, slightly different name)
    near_dup_src = [1, 8, 15, 25, 38, 50, 65, 80]
    near_dups = df[df["customer_id"].isin([f"U{3000+i}" for i in near_dup_src])].copy()
    near_dups["full_name"] = near_dups["full_name"].apply(lambda x: x + "_dup")
    df = pd.concat([df, near_dups], ignore_index=True)  # now 100 rows

    # 3. Inject 6 missing ages (use NaN)
    missing_age_idx = [2, 10, 23, 47, 68, 90]
    df.loc[missing_age_idx, "age"] = np.nan

    # 4. Inject 4 missing signup_dates
    missing_date_idx = [14, 35, 57, 83]
    df.loc[missing_date_idx, "signup_date"] = np.nan

    # 5. Inject 5 age outliers
    outlier_age_idx = [4, 19, 42, 64, 87]
    df.loc[outlier_age_idx, "age"] = [3, 105, 7, 110, 2]

    # 6. Inject inconsistent city casing
    bad_city_idx = [0, 6, 16, 29, 40, 53, 70, 84, 95]
    variants = list(CITY_VARIANTS.keys())
    for i, idx in enumerate(bad_city_idx):
        if idx < len(df):
            df.loc[idx, "city"] = variants[i % len(variants)]

    # 7. Inject invalid emails
    invalid_email_idx = [7, 21, 37, 55, 77]
    bad_emails = ["not-an-email", "missing_at_sign.com", "no@", "@nodomain", "spaces in@email.com"]
    for i, idx in enumerate(invalid_email_idx):
        if idx < len(df):
            df.loc[idx, "email"] = bad_emails[i]

    return df.reset_index(drop=True)


_SEED_DF = _make_dataset()

EXACT_DUP_IDX       = set(range(85, 92))        # rows 85–91 are exact dups
NEAR_DUP_IDX        = set(range(92, 100))        # rows 92–99 are near-dups
MISSING_AGE_IDX     = {2, 10, 23, 47, 68, 90}
MISSING_DATE_IDX    = {14, 35, 57, 83}
OUTLIER_AGE_IDX     = {4, 19, 42, 64, 87}
INVALID_EMAIL_IDX   = {7, 21, 37, 55, 77}
ALL_BAD_IDX         = (EXACT_DUP_IDX | NEAR_DUP_IDX | OUTLIER_AGE_IDX | INVALID_EMAIL_IDX)


def get_initial_df() -> pd.DataFrame:
    return _SEED_DF.copy()


def get_metadata() -> Dict[str, Any]:
    return {
        "task_id": "task3_full_pipeline",
        "name": "Full Data Cleaning Pipeline",
        "difficulty": "hard",
        "max_steps": 30,
        "description": (
            "A 100-row CRM export needs a full cleaning pass. "
            "Issues: (1) 7 exact duplicate rows + 8 near-duplicate customer records; "
            "(2) 6 missing ages (fill with median) + 4 missing signup_dates (fill with mode); "
            "(3) 5 unrealistic age values (< 10 or > 100) — remove those rows; "
            "(4) inconsistent city casing — standardise to Title Case; "
            "(5) 5 invalid email addresses — filter those rows; "
            "(6) `purchase_amount` stored as '$XX.XX' strings — convert to float."
        ),
        "ground_truth": {
            "exact_duplicates": 7,
            "near_duplicates": 8,
            "missing_ages": 6,
            "missing_dates": 4,
            "age_outliers": 5,
            "invalid_emails": 5,
            "purchase_dtype": "float64",
            "canonical_cities": sorted(CANONICAL_CITIES),
        },
    }


# ── Step reward ──────────────────────────────────────────────────────────────

def step_reward(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> Tuple[float, str]:
    reward = 0.0
    msgs   = []

    # Duplicate reduction
    prev_dups = prev_df.duplicated().sum()
    curr_dups = curr_df.duplicated().sum()
    if curr_dups < prev_dups:
        credit = (prev_dups - curr_dups) / (len(EXACT_DUP_IDX) + len(NEAR_DUP_IDX)) * 0.12
        reward += credit
        msgs.append(f"Dups removed +{credit:.3f}")

    # Null reduction (age + date)
    p_null = prev_df[["age","signup_date"]].isna().sum().sum() if all(c in prev_df for c in ["age","signup_date"]) else 0
    c_null = curr_df[["age","signup_date"]].isna().sum().sum() if all(c in curr_df for c in ["age","signup_date"]) else 0
    if c_null < p_null:
        credit = (p_null - c_null) / (len(MISSING_AGE_IDX) + len(MISSING_DATE_IDX)) * 0.10
        reward += credit
        msgs.append(f"Nulls filled +{credit:.3f}")

    # City standardisation
    if "city" in curr_df.columns:
        p_bad = sum(1 for v in prev_df.get("city",[]) if v not in CANONICAL_CITIES)
        c_bad = sum(1 for v in curr_df["city"] if v not in CANONICAL_CITIES)
        if c_bad < p_bad:
            credit = (p_bad - c_bad) / 9 * 0.08
            reward += credit
            msgs.append(f"Cities fixed +{credit:.3f}")

    # dtype purchase
    p_numeric = pd.api.types.is_numeric_dtype(prev_df.get("purchase_amount", pd.Series(dtype=object)))
    c_numeric = pd.api.types.is_numeric_dtype(curr_df.get("purchase_amount", pd.Series(dtype=object)))
    if not p_numeric and c_numeric:
        reward += 0.12
        msgs.append("purchase_amount dtype fix +0.12")

    return round(float(reward), 4), " | ".join(msgs) or "No measurable change."


# ── Final grader ─────────────────────────────────────────────────────────────

def grade(final_df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, float] = {}
    fb: list[str] = []

    n = len(final_df)

    # 1. Deduplication
    exact_remain = final_df.duplicated().sum()
    dedup_score  = 1.0 - min(1.0, exact_remain / max(1, len(EXACT_DUP_IDX)))
    # Also penalise if near-dups on customer_id remain
    if "customer_id" in final_df.columns:
        cid_dups = final_df["customer_id"].duplicated().sum()
        near_score = 1.0 - min(1.0, cid_dups / max(1, len(NEAR_DUP_IDX)))
    else:
        near_score = 0.0
    results["deduplication"] = round((dedup_score * 0.5 + near_score * 0.5), 3)
    fb.append(f"dedup={results['deduplication']:.2f} ({exact_remain} exact, {cid_dups if 'customer_id' in final_df else '?'} cid dups remain)")

    # 2. Missing values filled
    age_null  = final_df["age"].isna().sum()  if "age"         in final_df else len(MISSING_AGE_IDX)
    date_null = final_df["signup_date"].isna().sum() if "signup_date" in final_df else len(MISSING_DATE_IDX)
    filled    = (len(MISSING_AGE_IDX) - age_null) + (len(MISSING_DATE_IDX) - date_null)
    miss_score = filled / (len(MISSING_AGE_IDX) + len(MISSING_DATE_IDX))
    results["missing_filled"] = round(max(0.0, float(miss_score)), 3)
    fb.append(f"missing_filled={results['missing_filled']:.2f}")

    # 3. Age validity
    if "age" in final_df.columns:
        invalid_ages = ((final_df["age"] < 10) | (final_df["age"] > 100)).sum()
        age_valid    = 1.0 - min(1.0, invalid_ages / max(1, len(OUTLIER_AGE_IDX)))
    else:
        age_valid = 0.0
    results["age_validity"] = round(float(age_valid), 3)
    fb.append(f"age_validity={age_valid:.2f}")

    # 4. City standardisation
    if "city" in final_df.columns:
        bad_city = sum(1 for v in final_df["city"] if str(v) not in CANONICAL_CITIES)
        city_score = 1.0 - min(1.0, bad_city / 9)
    else:
        city_score = 0.0
    results["city_standard"] = round(float(city_score), 3)
    fb.append(f"city_standard={city_score:.2f}")

    # 5. Email validity
    if "email" in final_df.columns:
        bad_email_remain = sum(1 for e in final_df["email"] if not _valid_email(str(e)))
        email_score      = 1.0 - min(1.0, bad_email_remain / max(1, len(INVALID_EMAIL_IDX)))
    else:
        email_score = 0.0
    results["email_validity"] = round(float(email_score), 3)
    fb.append(f"email_validity={email_score:.2f}")

    # 6. purchase_amount dtype
    pa_numeric = (
        "purchase_amount" in final_df.columns
        and pd.api.types.is_numeric_dtype(final_df["purchase_amount"])
    )
    results["dtype_purchase"] = 1.0 if pa_numeric else 0.0
    fb.append(f"dtype_purchase={'PASS' if pa_numeric else 'FAIL'}")

    score = (
        0.20 * results["deduplication"]
        + 0.15 * results["missing_filled"]
        + 0.15 * results["age_validity"]
        + 0.15 * results["city_standard"]
        + 0.20 * results["email_validity"]
        + 0.15 * results["dtype_purchase"]
    )
    score = round(float(score), 4)

    return {
        "score":     score,
        "breakdown": results,
        "feedback":  " | ".join(fb),
        "passed":    score >= 0.5,
    }
