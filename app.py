from __future__ import annotations
import os, json, traceback
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from env import DataCleaningEnvironment, Action
from env.models import (
    GraderRequest, GraderResponse,
    TaskInfo, TaskListResponse,
    BaselineTaskResult, BaselineResponse,
    StepResult,
)
from env.tasks import TASK_REGISTRY

# ── App setup ─────────────────────────────────────────

app = FastAPI(
    title="Customer Data Cleaning — OpenEnv",
    version="1.0.0",
)

_env = DataCleaningEnvironment()
DEFAULT_TASK = "task1_missing_values"

@app.on_event("startup")
async def startup():
    _env.reset(task_id=DEFAULT_TASK)

# ── Root ─────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

# ── Health ───────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

# ── RESET (FINAL FIXED VERSION) ──────────────────────

from fastapi import Body
@app.post("/reset")
def reset():
    obs = _env.reset(task_id="task1_missing_values")
    return obs.model_dump()

# ── STEP ─────────────────────────────────────────────

@app.post("/step")
async def step(request: Request):
    try:
        body = await request.body()

        if not body:
            raise HTTPException(status_code=400, detail="Action body required")

        data = json.loads(body)
        action = Action(**data)

        result: StepResult = _env.step(action)
        return result.model_dump()

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
# ── STATE ────────────────────────────────────────────

@app.get("/state")
def state():
    return _env.state()

# ── TASKS ────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    task_list = []
    for tid in TASK_REGISTRY:
        meta = TASK_REGISTRY[tid].get_metadata()
        task_list.append(TaskInfo(
            task_id=tid,
            name=meta["name"],
            difficulty=meta["difficulty"],
            description=meta["description"],
            max_steps=meta["max_steps"],
            action_schema=Action.model_json_schema()
        ))
    return TaskListResponse(tasks=task_list).model_dump()

# ── GRADER ───────────────────────────────────────────

@app.post("/grader")
async def grader(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)

        req = GraderRequest(**data)

        if req.task_id not in TASK_REGISTRY:
            raise HTTPException(status_code=400, detail="Invalid task_id")

        df = pd.DataFrame(req.final_data)
        result = TASK_REGISTRY[req.task_id].grade(df)

        return GraderResponse(
            task_id=req.task_id,
            score=result["score"],
            breakdown=result["breakdown"],
            feedback=result["feedback"],
            passed=result["passed"],
        ).model_dump()

    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

# ── BASELINE ─────────────────────────────────────────

@app.post("/baseline")
def baseline():
    try:
        from baseline import run_baseline_task

        results: List[BaselineTaskResult] = []

        for task_id in TASK_REGISTRY:
            r = run_baseline_task(task_id)
            results.append(BaselineTaskResult(**r))

        mean_score = round(sum(r.score for r in results) / len(results), 4)

        return BaselineResponse(
            model="rule-based",
            results=results,
            mean_score=mean_score,
        ).model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── RUN ──────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)