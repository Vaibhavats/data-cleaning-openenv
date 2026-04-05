"""
Task 2 (Medium): Fix Outliers and Data Types
=============================================
A 60-row sales dataset has three compounding problems:
  1. `age` column is stored as strings (object dtype) → must be cast to int
  2. `salary` has 5 extreme outliers (> 3 std deviations above mean)
  3. `purchase_amount` has 4 negative values (invalid in context)

Grader checks:
  - dtype of `age` is numeric
  - salary outliers removed (precision/recall on known indices)
  - no negative purchase_amount rows remain
  - minimal valid-row loss (preservation)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, Set, Tuple

# ── Seed dataset ────────────────────────────────────────────────────────────

def _make_dataset() -> pd.DataFrame:
    np.random.seed(7)
    n = 60

    # Normal salary range 40k–90k
    salaries = np.round(np.random.normal(62000, 10000, size=n), 2)
    salaries = np.clip(salaries, 35000, 90000)

    # Inject 5 salary outliers (> 3 SD above mean ≈ 92000)
    outlier_salary_idx = [4, 14, 28, 41, 55]
    for i in outlier_salary_idx:
        salaries[i] = np.random.uniform(130000, 180000)

    # Normal purchases
    purchases = np.round(np.random.uniform(10, 500, size=n), 2)
    # Inject 4 negative purchase amounts
    negative_purchase_idx = [6, 20, 33, 50]
    for i in negative_purchase_idx:
        purchases[i] = np.round(-np.random.uniform(50, 200), 2)

    df = pd.DataFrame({
        "sale_id":        [f"S{2000+i}" for i in range(n)],
        "rep_name":       [f"Rep_{i}" for i in range(n)],
        "age":            pd.array([str(int(a)) for a in np.random.randint(21, 60, size=n)], dtype=object),  # stored as str!
        "salary":         salaries,
        "purchase_amount": purchases,
        "region":         np.random.choice(["North","South","East","West"], size=n),
    })
    return df


_SEED_DF              = _make_dataset()
OUTLIER_SALARY_IDX    = {4, 14, 28, 41, 55}
NEGATIVE_PURCHASE_IDX = {6, 20, 33, 50}

# Threshold used for grading outlier removal
_SALARY_MEAN  = float(_SEED_DF["salary"].drop(index=list(OUTLIER_SALARY_IDX)).mean())
_SALARY_STD   = float(_SEED_DF["salary"].drop(index=list(OUTLIER_SALARY_IDX)).std())
OUTLIER_THRESHOLD = _SALARY_MEAN + 3 * _SALARY_STD


def get_initial_df() -> pd.DataFrame:
    return _SEED_DF.copy()


def get_metadata() -> Dict[str, Any]:
    return {
        "task_id": "task2_outliers_dtype",
        "name": "Fix Outliers and Data Types",
        "difficulty": "medium",
        "max_steps": 20,
        "description": (
            "A 60-row sales dataset has three issues: "
            "(1) `age` is stored as strings — cast it to int or float. "
            "(2) `salary` contains 5 extreme outliers above 3 standard deviations "
            "from the mean — remove those rows. "
            "(3) `purchase_amount` has 4 negative values — filter them out."
        ),
        "ground_truth": {
            "age_target_dtype": "int64",
            "outlier_count": len(OUTLIER_SALARY_IDX),
            "salary_outlier_threshold": round(OUTLIER_THRESHOLD, 2),
            "negative_purchase_count": len(NEGATIVE_PURCHASE_IDX),
        },
    }


# ── Step-level reward ────────────────────────────────────────────────────────

def step_reward(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> Tuple[float, str]:
    """Dense reward: credit each distinct improvement incrementally."""
    reward = 0.0
    msgs   = []

    # dtype fix
    prev_age_numeric = pd.api.types.is_numeric_dtype(prev_df.get("age", pd.Series(dtype=object)))
    curr_age_numeric = pd.api.types.is_numeric_dtype(curr_df.get("age", pd.Series(dtype=object)))
    if not prev_age_numeric and curr_age_numeric:
        reward += 0.15
        msgs.append("dtype fix credited (+0.15)")

    # negative purchase removal
    prev_neg = (prev_df["purchase_amount"] < 0).sum() if "purchase_amount" in prev_df.columns else 0
    curr_neg = (curr_df["purchase_amount"] < 0).sum() if "purchase_amount" in curr_df.columns else 0
    removed_neg = max(0, prev_neg - curr_neg)
    if removed_neg:
        credit = removed_neg / len(NEGATIVE_PURCHASE_IDX) * 0.15
        reward += credit
        msgs.append(f"{removed_neg} negative purchase(s) removed (+{credit:.3f})")

    # outlier removal (heuristic: salary > threshold)
    if "salary" in prev_df.columns and "salary" in curr_df.columns:
        prev_out = (prev_df["salary"] > OUTLIER_THRESHOLD).sum()
        curr_out = (curr_df["salary"] > OUTLIER_THRESHOLD).sum()
        removed_out = max(0, prev_out - curr_out)
        if removed_out:
            credit = removed_out / len(OUTLIER_SALARY_IDX) * 0.15
            reward += credit
            msgs.append(f"{removed_out} salary outlier(s) removed (+{credit:.3f})")

    # Penalise if too many valid rows were deleted
    valid_prev = len(prev_df) - (
        int((prev_df.get("salary", pd.Series()) > OUTLIER_THRESHOLD).sum())
        + int((prev_df.get("purchase_amount", pd.Series()) < 0).sum())
    )
    valid_curr = len(curr_df) - (
        int((curr_df.get("salary", pd.Series()) > OUTLIER_THRESHOLD).sum())
        + int((curr_df.get("purchase_amount", pd.Series()) < 0).sum())
    )
    over_deletion = max(0, valid_prev - valid_curr)
    if over_deletion:
        penalty = over_deletion * 0.03
        reward  = max(0.0, reward - penalty)
        msgs.append(f"{over_deletion} valid rows deleted (penalty -{penalty:.2f})")

    return round(float(reward), 4), " | ".join(msgs) or "No measurable improvement."


# ── Final grader ─────────────────────────────────────────────────────────────

def grade(final_df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, float] = {}
    feedback_parts: list[str] = []

    # 1. dtype_fix
    age_ok = "age" in final_df and pd.api.types.is_numeric_dtype(final_df["age"])
    results["dtype_fix"] = 1.0 if age_ok else 0.0
    feedback_parts.append(
        f"dtype_fix={'PASS' if age_ok else 'FAIL'}: "
        f"`age` dtype is {final_df['age'].dtype if 'age' in final_df else 'missing'}."
    )

    # 2. Outlier removal — precision/recall on known indices
    if "salary" in final_df.columns:
        # Which original indices are still in the dataframe?
        # We use the fact that sale_id encodes the original index.
        try:
            # sale_id format is "S2000", "S2001", … "S2059" → index = int - 2000
            remaining_ids = set(int(str(sid).lstrip("S")) - 2000 for sid in final_df["sale_id"])
            remaining_outlier_idx = OUTLIER_SALARY_IDX & {i for i in remaining_ids if 0 <= i < 60}
            tp = len(OUTLIER_SALARY_IDX) - len(remaining_outlier_idx)   # correctly removed
            fp_proxy = max(0, (60 - len(OUTLIER_SALARY_IDX)) - (len(final_df) - tp))
            precision  = tp / max(1, tp + fp_proxy)
            recall     = tp / len(OUTLIER_SALARY_IDX)
            f1 = (2 * precision * recall / max(1e-9, precision + recall))
        except Exception:
            # Fallback: count rows still above threshold
            remaining_out = (final_df["salary"] > OUTLIER_THRESHOLD).sum()
            f1 = 1.0 - remaining_out / len(OUTLIER_SALARY_IDX)
            f1 = max(0.0, f1)
    else:
        f1 = 0.0
    results["outlier_removal"] = round(float(f1), 3)
    feedback_parts.append(f"outlier_removal F1={f1:.2f}")

    # 3. Negative purchase removal
    if "purchase_amount" in final_df.columns:
        neg_remain = (final_df["purchase_amount"] < 0).sum()
        neg_score  = 1.0 - neg_remain / len(NEGATIVE_PURCHASE_IDX)
        neg_score  = max(0.0, neg_score)
    else:
        neg_score = 0.0
    results["negative_purchase"] = round(float(neg_score), 3)
    feedback_parts.append(
        f"negative_purchase={neg_score:.2f}: "
        f"{(final_df['purchase_amount'] < 0).sum() if 'purchase_amount' in final_df.columns else '?'} negatives remain."
    )

    # 4. Preservation — should keep ~51 valid rows
    expected_valid = 60 - len(OUTLIER_SALARY_IDX) - len(NEGATIVE_PURCHASE_IDX)  # 51
    preservation   = min(len(final_df), expected_valid) / expected_valid
    results["preservation"] = round(float(preservation), 3)
    feedback_parts.append(f"preservation={preservation:.2f}: {len(final_df)} rows remain.")

    score = (
        0.25 * results["dtype_fix"]
        + 0.35 * results["outlier_removal"]
        + 0.25 * results["negative_purchase"]
        + 0.15 * results["preservation"]
    )
    score = round(float(score), 4)

    return {
        "score":     score,
        "breakdown": results,
        "feedback":  " | ".join(feedback_parts),
        "passed":    score >= 0.5,
    }
