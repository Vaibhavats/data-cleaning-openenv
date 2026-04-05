"""
DataCleaningEnvironment — core OpenEnv implementation.
Implements reset() / step() / state() with full typed I/O.
"""
from __future__ import annotations
import copy, json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .models import Action, Observation, Reward, StepResult, DatasetState
from .tasks import TASK_REGISTRY


# ── Helpers ─────────────────────────────────────────────────────────────────

def _df_to_state(df: pd.DataFrame) -> DatasetState:
    """Serialise a DataFrame to a DatasetState snapshot (cap at 200 rows)."""
    sample = df.head(200)
    stats: Dict[str, Any] = {}
    for col in df.columns:
        col_stats: Dict[str, Any] = {"dtype": str(df[col].dtype)}
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "mean":  round(float(df[col].mean()), 4) if df[col].notna().any() else None,
                "std":   round(float(df[col].std()),  4) if df[col].notna().any() else None,
                "min":   float(df[col].min()) if df[col].notna().any() else None,
                "max":   float(df[col].max()) if df[col].notna().any() else None,
            })
        else:
            col_stats["unique"] = int(df[col].nunique())
            top = df[col].value_counts().head(3).to_dict()
            col_stats["top_values"] = {str(k): int(v) for k, v in top.items()}
        stats[col] = col_stats

    return DatasetState(
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        shape=[len(df), len(df.columns)],
        data=json.loads(sample.to_json(orient="records", date_format="iso")),
        missing_counts={c: int(df[c].isna().sum()) for c in df.columns},
        stats=stats,
    )


def _apply_action(df: pd.DataFrame, action: Action) -> Tuple[pd.DataFrame, str, bool]:
    """
    Apply an action to the DataFrame.
    Returns (new_df, message, is_valid).
    Raises ValueError for malformed params.
    """
    atype  = action.action_type
    params = action.params
    df     = df.copy()

    if atype == "fill_missing":
        col      = params.get("column")
        strategy = params.get("strategy", "mean")
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False

        if strategy == "mean":
            fill_val = df[col].mean()
        elif strategy == "median":
            fill_val = df[col].median()
        elif strategy == "mode":
            fill_val = df[col].mode()[0] if not df[col].mode().empty else None
        elif strategy == "value":
            fill_val = params.get("value")
        else:
            return df, f"Unknown strategy '{strategy}'.", False

        if fill_val is None:
            return df, "Could not compute fill value.", False

        before = df[col].isna().sum()
        df[col] = df[col].fillna(fill_val)
        after  = df[col].isna().sum()
        return df, f"Filled {before - after} nulls in '{col}' with {strategy}={fill_val:.4g}.", True

    elif atype == "remove_outliers":
        col       = params.get("column")
        method    = params.get("method", "zscore")
        threshold = float(params.get("threshold", 3.0))
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, f"Column '{col}' is not numeric.", False

        before = len(df)
        if method == "zscore":
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z <= threshold]
        elif method == "iqr":
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            df     = df[(df[col] >= Q1 - threshold * IQR) & (df[col] <= Q3 + threshold * IQR)]
        else:
            return df, f"Unknown method '{method}'.", False

        removed = before - len(df)
        return df.reset_index(drop=True), f"Removed {removed} outlier rows from '{col}'.", True

    elif atype == "fix_dtype":
        col         = params.get("column")
        target_type = params.get("target_type", "float")
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False
        try:
            if target_type in ("int", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif target_type in ("float", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "str":
                df[col] = df[col].astype(str)
            elif target_type == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif target_type == "strip_and_float":
                # strip currency symbols before converting
                df[col] = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                return df, f"Unsupported target_type '{target_type}'.", False
        except Exception as exc:
            return df, f"dtype conversion failed: {exc}", False
        return df, f"Converted '{col}' to {target_type}.", True

    elif atype == "drop_duplicates":
        subset = params.get("subset", None)
        before = len(df)
        if subset:
            invalid_cols = [c for c in subset if c not in df.columns]
            if invalid_cols:
                return df, f"Columns not found: {invalid_cols}", False
            df = df.drop_duplicates(subset=subset)
        else:
            df = df.drop_duplicates()
        removed = before - len(df)
        return df.reset_index(drop=True), f"Dropped {removed} duplicate rows.", True

    elif atype == "normalize":
        col    = params.get("column")
        method = params.get("method", "minmax")
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, f"Column '{col}' is not numeric.", False
        if method == "minmax":
            mn, mx = df[col].min(), df[col].max()
            df[col] = (df[col] - mn) / (mx - mn) if mx != mn else 0.0
        elif method == "zscore":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            return df, f"Unknown normalisation method '{method}'.", False
        return df, f"Normalised '{col}' with {method}.", True

    elif atype == "filter_rows":
        col      = params.get("column")
        operator = params.get("operator", "gt")
        value    = params.get("value")
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False
        before = len(df)
        try:
            if operator == "gt":
                df = df[df[col] > value]
            elif operator == "lt":
                df = df[df[col] < value]
            elif operator == "gte":
                df = df[df[col] >= value]
            elif operator == "lte":
                df = df[df[col] <= value]
            elif operator == "eq":
                df = df[df[col] == value]
            elif operator == "ne":
                df = df[df[col] != value]
            elif operator == "contains":
                df = df[df[col].astype(str).str.contains(str(value), na=False)]
            elif operator == "not_contains":
                df = df[~df[col].astype(str).str.contains(str(value), na=False)]
            else:
                return df, f"Unknown operator '{operator}'.", False
        except Exception as exc:
            return df, f"filter_rows failed: {exc}", False
        removed = before - len(df)
        return df.reset_index(drop=True), f"Filtered {removed} rows where '{col}' {operator} {value}.", True

    elif atype == "map_values":
        col     = params.get("column")
        mapping = params.get("mapping", {})
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False
        if not isinstance(mapping, dict):
            return df, "params.mapping must be a dict.", False
        before_unique = df[col].nunique()
        df[col] = df[col].replace(mapping)
        after_unique = df[col].nunique()
        replaced = sum(1 for k in mapping if k in df[col].values or True)
        return df, f"Mapped values in '{col}' using {len(mapping)} rules.", True

    elif atype == "standardize_text":
        col   = params.get("column")
        case  = params.get("case", "title")
        strip = params.get("strip", True)
        if col not in df.columns:
            return df, f"Column '{col}' not found.", False
        s = df[col].astype(str)
        if strip:
            s = s.str.strip()
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        df[col] = s
        return df, f"Standardised text in '{col}' (case={case}, strip={strip}).", True

    elif atype == "submit":
        return df, "Episode submitted for grading.", True

    else:
        return df, f"Unknown action_type '{atype}'.", False


# ── Environment class ────────────────────────────────────────────────────────

class DataCleaningEnvironment:
    """
    OpenEnv-compliant environment for customer data cleaning tasks.
    Maintains episode state; thread-safety is the caller's responsibility.
    """

    MAX_STEP_PENALTY = 0.01  # small penalty per wasted step

    def __init__(self) -> None:
        self._task_id:      Optional[str]       = None
        self._task_module:  Any                  = None
        self._df:           Optional[pd.DataFrame] = None
        self._step_count:   int                  = 0
        self._max_steps:    int                  = 20
        self._done:         bool                 = False
        self._history:      list                 = []
        self._reward_accum: float                = 0.0

    # ── Public OpenEnv methods ───────────────────────────────────────────────

    def reset(self, task_id: str = "task1_missing_values") -> Observation:
        """Start a fresh episode for the given task."""
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_REGISTRY)}")

        self._task_id     = task_id
        self._task_module = TASK_REGISTRY[task_id]
        meta              = self._task_module.get_metadata()
        self._df          = self._task_module.get_initial_df()
        self._step_count  = 0
        self._max_steps   = meta["max_steps"]
        self._done        = False
        self._history     = []
        self._reward_accum = 0.0

        return self._build_obs(hints=["Episode started. Inspect the dataset and begin cleaning."])

    def step(self, action: Action) -> StepResult:
        """Apply an action and return (observation, reward, done, info)."""
        if self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        prev_df = self._df.copy()

        # Handle submit
        if action.action_type == "submit":
            return self._finalise()

        # Apply action
        new_df, msg, valid = _apply_action(self._df, action)
        self._df           = new_df
        self._step_count  += 1

        # Compute step reward
        step_r, r_msg = self._task_module.step_reward(prev_df, self._df)
        if not valid:
            step_r = -0.02
        self._reward_accum = round(min(1.0, self._reward_accum + step_r), 4)

        self._history.append({
            "step":   self._step_count,
            "action": action.model_dump(),
            "valid":  valid,
            "msg":    msg,
            "reward": step_r,
        })

        done = self._step_count >= self._max_steps
        if done:
            return self._finalise()

        hints = []
        if not valid:
            hints.append(f"Invalid action: {msg}")
        if step_r <= 0 and valid:
            hints.append("No measurable improvement — check columns or parameters.")

        reward_obj = Reward(
            score=step_r,
            breakdown={"step_reward": step_r},
            feedback=f"{msg} | {r_msg}",
        )
        return StepResult(
            observation=self._build_obs(hints=hints or None),
            reward=reward_obj,
            done=False,
            info={"valid": valid, "message": msg},
        )

    def state(self) -> Dict[str, Any]:
        """Return the raw environment state (useful for debugging)."""
        if self._df is None:
            return {"status": "not_started"}
        return {
            "task_id":       self._task_id,
            "step_count":    self._step_count,
            "max_steps":     self._max_steps,
            "done":          self._done,
            "reward_so_far": self._reward_accum,
            "shape":         list(self._df.shape),
            "missing_counts":{c: int(self._df[c].isna().sum()) for c in self._df.columns},
            "dtypes":        {c: str(self._df[c].dtype) for c in self._df.columns},
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_obs(self, hints=None) -> Observation:
        meta = self._task_module.get_metadata()
        return Observation(
            task_id=self._task_id,
            task_description=meta["description"],
            step_count=self._step_count,
            dataset=_df_to_state(self._df),
            action_history=self._history[-10:],  # last 10 actions
            reward_so_far=self._reward_accum,
            hints=hints,
        )

    def _finalise(self) -> StepResult:
        """Run the official grader and close the episode."""
        self._done = True
        result     = self._task_module.grade(self._df)
        final_r    = Reward(
            score=result["score"],
            breakdown=result["breakdown"],
            feedback=result["feedback"],
        )
        self._history.append({"step": self._step_count, "action": "submit", "final_grade": result})
        return StepResult(
            observation=self._build_obs(hints=["Episode complete."]),
            reward=final_r,
            done=True,
            info={
                "final_score": result["score"],
                "passed":      result["passed"],
                "breakdown":   result["breakdown"],
            },
        )
