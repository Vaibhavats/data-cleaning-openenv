"""
Typed Pydantic models for the Customer Data Cleaning OpenEnv environment.
Implements: Observation, Action, Reward, StepResult, and auxiliary models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ── Dataset snapshot ────────────────────────────────────────────────────────

class DatasetState(BaseModel):
    """A serialisable snapshot of the current (possibly dirty) DataFrame."""
    columns: List[str] = Field(description="Column names in order")
    dtypes: Dict[str, str] = Field(description="Column → dtype string, e.g. 'float64'")
    shape: List[int] = Field(description="[rows, cols]")
    data: List[Dict[str, Any]] = Field(description="Rows as list of dicts (≤200 rows shown)")
    missing_counts: Dict[str, int] = Field(description="Column → count of nulls")
    stats: Dict[str, Any] = Field(
        description="Per-column descriptive statistics (mean, std, min, max, unique)"
    )


# ── Core OpenEnv models ─────────────────────────────────────────────────────

class Observation(BaseModel):
    """Returned by reset() and step()."""
    task_id: str
    task_description: str
    step_count: int = Field(ge=0)
    dataset: DatasetState
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    reward_so_far: float = Field(ge=0.0, le=1.0, default=0.0)
    hints: Optional[List[str]] = Field(
        default=None,
        description="Optional natural-language hints surfaced after bad actions"
    )


class Action(BaseModel):
    """
    Sent by the agent to step().
    action_type selects which cleaning operation to perform;
    params carries operation-specific arguments (see README for full schema).
    """
    action_type: Literal[
        "fill_missing",
        "remove_outliers",
        "fix_dtype",
        "drop_duplicates",
        "normalize",
        "filter_rows",
        "standardize_text",
        "map_values",
        "validate_email",
        "submit",
    ] = Field(description="Cleaning operation to apply")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Operation parameters. Examples:\n"
            "  fill_missing  → {column, strategy: mean|median|mode|value, value?}\n"
            "  remove_outliers → {column, method: iqr|zscore, threshold: float}\n"
            "  fix_dtype     → {column, target_type: int|float|str|datetime}\n"
            "  drop_duplicates → {subset?: [col,...]}\n"
            "  normalize     → {column, method: minmax|zscore}\n"
            "  filter_rows   → {column, operator: gt|lt|eq|ne|contains, value}\n"
            "  standardize_text → {column, case: lower|upper|title, strip: bool}\n"
            "  map_values       → {column, mapping: {old_val: new_val, ...}}\n"
            "  submit        → {}  # finalises episode and triggers grader"
        ),
    )


class Reward(BaseModel):
    """Returned inside StepResult; full grader report on submit."""
    score: float = Field(ge=0.0, le=1.0, description="Scalar reward [0, 1]")
    breakdown: Dict[str, float] = Field(
        description="Per-dimension sub-scores, e.g. {'missing': 0.9, 'outliers': 0.7}"
    )
    feedback: str = Field(description="Human-readable explanation of the reward")


class StepResult(BaseModel):
    """Full response from step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ── Task metadata ────────────────────────────────────────────────────────────

class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    action_schema: Dict[str, Any]


class TaskListResponse(BaseModel):
    tasks: List[TaskInfo]


# ── Grader request/response ─────────────────────────────────────────────────

class GraderRequest(BaseModel):
    task_id: str
    final_data: List[Dict[str, Any]] = Field(
        description="The agent's final dataset rows (post-cleaning)"
    )


class GraderResponse(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str
    passed: bool = Field(description="True if score >= 0.5")


# ── Baseline result ──────────────────────────────────────────────────────────

class BaselineTaskResult(BaseModel):
    task_id: str
    score: float
    steps_taken: int
    breakdown: Dict[str, float]


class BaselineResponse(BaseModel):
    model: str
    results: List[BaselineTaskResult]
    mean_score: float
