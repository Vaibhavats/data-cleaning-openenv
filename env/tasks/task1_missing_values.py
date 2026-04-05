"""
Task 1 (Easy): Fill Missing Values
===================================
A 40-row customer table has intentional nulls in `age` (fill with median)
and `annual_income` (fill with mean). The agent must identify which columns
have missing values and apply the correct imputation strategy.

Grader checks:
  - All nulls filled (completeness score)
  - Correct imputation values used (accuracy score)
  - No unnecessary columns dropped or rows removed (preservation score)
"""
from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

# ── Seed dataset ────────────────────────────────────────────────────────────

def _make_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 40
    ages_full   = np.random.randint(22, 65, size=n).astype(float)
    incomes_full = np.round(np.random.normal(55000, 12000, size=n), 2)
    incomes_full = np.clip(incomes_full, 25000, 110000)

    df = pd.DataFrame({
        "customer_id": [f"C{1000+i}" for i in range(n)],
        "name":        [f"Customer_{i}" for i in range(n)],
        "email":       [f"customer{i}@example.com" for i in range(n)],
        "age":         ages_full,
        "annual_income": incomes_full,
        "city":        np.random.choice(
            ["New York","Los Angeles","Chicago","Houston","Phoenix"], size=n
        ),
    })

    # Inject exactly 6 missing ages (indices fixed for reproducibility)
    missing_age_idx = [3, 7, 12, 19, 27, 35]
    df.loc[missing_age_idx, "age"] = np.nan

    # Inject exactly 5 missing incomes
    missing_income_idx = [1, 9, 22, 30, 38]
    df.loc[missing_income_idx, "annual_income"] = np.nan

    return df


# Public ground-truth values (computed on the *non-null* rows)
_SEED_DF           = _make_dataset()
_TRUE_AGE_MEDIAN   = float(np.median(_SEED_DF["age"].dropna()))
_TRUE_INCOME_MEAN  = float(_SEED_DF["annual_income"].dropna().mean())

MISSING_AGE_IDX    = [3, 7, 12, 19, 27, 35]
MISSING_INCOME_IDX = [1, 9, 22, 30, 38]


def get_initial_df() -> pd.DataFrame:
    """Return a fresh copy of the dirty dataset."""
    return _SEED_DF.copy()


def get_metadata() -> Dict[str, Any]:
    return {
        "task_id": "task1_missing_values",
        "name": "Fill Missing Values",
        "difficulty": "easy",
        "max_steps": 15,
        "description": (
            "A 40-row customer dataset has 6 missing ages and 5 missing income values. "
            "Fill the missing `age` values using the column median and the missing "
            "`annual_income` values using the column mean. Do not remove any rows."
        ),
        "ground_truth": {
            "age_strategy": "median",
            "age_fill_value": _TRUE_AGE_MEDIAN,
            "income_strategy": "mean",
            "income_fill_value": round(_TRUE_INCOME_MEAN, 2),
            "missing_age_count": len(MISSING_AGE_IDX),
            "missing_income_count": len(MISSING_INCOME_IDX),
        },
    }


# ── Step-level reward ────────────────────────────────────────────────────────

def step_reward(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> Tuple[float, str]:
    """
    Immediate reward signal after an action.
    Rewards any reduction in nulls; penalises row deletion.
    """
    prev_nulls = prev_df[["age", "annual_income"]].isna().sum().sum()
    curr_nulls = curr_df[["age", "annual_income"]].isna().sum().sum()
    null_reduction = max(0, prev_nulls - curr_nulls)

    row_penalty = max(0, len(prev_df) - len(curr_df)) * 0.05

    reward = (null_reduction / (len(MISSING_AGE_IDX) + len(MISSING_INCOME_IDX))) * 0.4
    reward = max(0.0, reward - row_penalty)
    msg    = f"Nulls reduced by {null_reduction}. Row penalty: {row_penalty:.2f}."
    return round(float(reward), 4), msg


# ── Final grader ─────────────────────────────────────────────────────────────

def grade(final_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Score the agent's final dataset on three dimensions:
      1. completeness  – are all nulls filled?
      2. accuracy      – were the correct imputation values used?
      3. preservation  – did the agent keep all 40 rows?
    """
    # Edge case: empty dataframe
    if len(final_df) == 0:
        return {
            "score": 0.0,
            "breakdown": {"completeness": 0.0, "accuracy": 0.0, "preservation": 0.0},
            "feedback": "Empty dataframe submitted — all rows were deleted.",
            "passed": False,
        }

    results: Dict[str, float] = {}
    feedback_parts: list[str] = []

    # 1. Completeness
    age_nulls    = final_df["age"].isna().sum()          if "age"           in final_df else len(MISSING_AGE_IDX)
    income_nulls = final_df["annual_income"].isna().sum() if "annual_income" in final_df else len(MISSING_INCOME_IDX)
    total_missing = len(MISSING_AGE_IDX) + len(MISSING_INCOME_IDX)
    filled        = total_missing - (age_nulls + income_nulls)
    completeness  = filled / total_missing
    results["completeness"] = round(completeness, 3)
    feedback_parts.append(
        f"Completeness {completeness:.0%}: "
        f"{age_nulls} age nulls and {income_nulls} income nulls remain."
    )

    # 2. Accuracy – check filled values are within 1 unit of correct answer
    accuracy_scores = []
    if "age" in final_df:
        for idx in MISSING_AGE_IDX:
            if idx < len(final_df) and not pd.isna(final_df["age"].iloc[idx]):
                filled_val = float(final_df["age"].iloc[idx])
                tol        = 1.0
                accuracy_scores.append(1.0 if abs(filled_val - _TRUE_AGE_MEDIAN) <= tol else 0.0)
            else:
                accuracy_scores.append(0.0)

    if "annual_income" in final_df:
        for idx in MISSING_INCOME_IDX:
            if idx < len(final_df) and not pd.isna(final_df["annual_income"].iloc[idx]):
                filled_val = float(final_df["annual_income"].iloc[idx])
                tol        = 500.0
                accuracy_scores.append(1.0 if abs(filled_val - _TRUE_INCOME_MEAN) <= tol else 0.0)
            else:
                accuracy_scores.append(0.0)

    accuracy = float(np.mean(accuracy_scores)) if accuracy_scores else 0.0
    results["accuracy"] = round(accuracy, 3)
    feedback_parts.append(
        f"Accuracy {accuracy:.0%}: correct imputation value used for "
        f"{sum(1 for s in accuracy_scores if s==1.0)}/{len(accuracy_scores)} cells."
    )

    # 3. Preservation
    expected_rows = 40
    actual_rows   = len(final_df)
    preservation  = min(actual_rows, expected_rows) / expected_rows
    results["preservation"] = round(preservation, 3)
    feedback_parts.append(
        f"Preservation {preservation:.0%}: {actual_rows}/{expected_rows} rows retained."
    )

    # Weighted final score
    score = (
        0.40 * results["completeness"]
        + 0.40 * results["accuracy"]
        + 0.20 * results["preservation"]
    )
    score = round(float(score), 4)

    return {
        "score":     score,
        "breakdown": results,
        "feedback":  " | ".join(feedback_parts),
        "passed":    score >= 0.5,
    }
