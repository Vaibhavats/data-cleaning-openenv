"""
FastAPI server for the Customer Data Cleaning OpenEnv environment.

Endpoints:
  POST /reset        – start a new episode
  POST /step         – apply an action
  GET  /state        – current episode state
  GET  /tasks        – list tasks + action schema
  POST /grader       – standalone grader (accepts final_data)
  POST /baseline     – run the built-in baseline agent on all tasks
  GET  /health       – liveness check
"""
from __future__ import annotations
import os, json, time, traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import DataCleaningEnvironment, Action
from env.models import (
    GraderRequest, GraderResponse,
    TaskInfo, TaskListResponse,
    BaselineTaskResult, BaselineResponse,
    StepResult,
)
from env.tasks import TASK_REGISTRY

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Data Cleaning — OpenEnv",
    version="1.0.0",
    description=(
        "An RL environment where agents clean messy customer datasets. "
        "Implements the full OpenEnv spec: reset/step/state, typed models, "
        "3 graded tasks (easy→medium→hard)."
    ),
)

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# One global environment instance per server (single-session demo).
# For multi-agent use, replace with session-keyed dict + middleware.
_env = DataCleaningEnvironment()


# ── Request models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_missing_values"


# ── Helpers ──────────────────────────────────────────────────────────────────

ACTION_SCHEMA = {
    "action_type": {
        "type": "string",
        "enum": [
            "fill_missing", "remove_outliers", "fix_dtype",
            "drop_duplicates", "normalize", "filter_rows",
            "standardize_text", "map_values", "submit",
        ],
    },
    "params": {
        "type": "object",
        "description": "Operation-specific parameters",
        "examples": {
            "fill_missing":     {"column": "age", "strategy": "median"},
            "remove_outliers":  {"column": "salary", "method": "zscore", "threshold": 3.0},
            "fix_dtype":        {"column": "age", "target_type": "int"},
            "drop_duplicates":  {"subset": ["customer_id"]},
            "normalize":        {"column": "income", "method": "minmax"},
            "filter_rows":      {"column": "purchase_amount", "operator": "gt", "value": 0},
            "standardize_text": {"column": "city", "case": "title", "strip": True},
            "map_values":       {"column": "city", "mapping": {"LA": "Los Angeles", "NYC": "New York"}},
            "submit":           {},
        },
    },
}


def _task_info(task_id: str) -> TaskInfo:
    meta = TASK_REGISTRY[task_id].get_metadata()
    return TaskInfo(
        task_id=task_id,
        name=meta["name"],
        difficulty=meta["difficulty"],
        description=meta["description"],
        max_steps=meta["max_steps"],
        action_schema=ACTION_SCHEMA,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "customer-data-cleaning", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Reset the environment and return the initial observation."""
    try:
        obs = _env.reset(task_id=req.task_id)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/step")
def step(action: Action):
    """Apply an action and return (observation, reward, done, info)."""
    try:
        result: StepResult = _env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
def state():
    """Return the raw environment state."""
    return _env.state()


@app.get("/tasks")
def list_tasks():
    """Return all tasks with their metadata and action schema."""
    tasks = [_task_info(tid) for tid in TASK_REGISTRY]
    return TaskListResponse(tasks=tasks).model_dump()


@app.post("/grader")
def grader(req: GraderRequest):
    """
    Standalone grader: submit a final dataset and receive a score.
    Useful for offline evaluation without running a full episode.
    """
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'")
    try:
        df     = pd.DataFrame(req.final_data)
        result = TASK_REGISTRY[req.task_id].grade(df)
        return GraderResponse(
            task_id=req.task_id,
            score=result["score"],
            breakdown=result["breakdown"],
            feedback=result["feedback"],
            passed=result["passed"],
        ).model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/baseline")
def baseline():
    """
    Run the built-in rule-based baseline agent on all 3 tasks.
    Returns per-task scores and mean score.
    This endpoint mirrors the baseline.py script for quick validation.
    """
    from baseline import run_baseline_task
    results: List[BaselineTaskResult] = []
    for task_id in TASK_REGISTRY:
        r = run_baseline_task(task_id)
        results.append(BaselineTaskResult(**r))
    mean_score = round(sum(r.score for r in results) / len(results), 4)
    return BaselineResponse(
        model="rule-based-baseline",
        results=results,
        mean_score=mean_score,
    ).model_dump()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
