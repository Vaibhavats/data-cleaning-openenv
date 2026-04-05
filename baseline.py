"""
baseline.py — Baseline inference script for the Customer Data Cleaning OpenEnv.

Two modes:
  1. Rule-based baseline (default, no API key needed):
     python baseline.py

  2. LLM baseline using OpenAI-compatible API:
     OPENAI_API_KEY=<key> OPENAI_BASE_URL=<url> python baseline.py --llm
     (Works with OpenAI, Together AI, Groq, local vLLM, etc.)

Produces reproducible scores on all 3 tasks.
"""
from __future__ import annotations
import os, sys, json, time, argparse, textwrap
from typing import Any, Dict, List

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL         = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL        = os.environ.get("LLM_MODEL", "gpt-4o-mini")

TASK_IDS = ["task1_missing_values", "task2_outliers_dtype", "task3_full_pipeline"]


# ── Rule-based strategy per task ─────────────────────────────────────────────

def _rule_actions_task1(obs: Dict) -> List[Dict]:
    """Fill missing age with median, income with mean, then submit."""
    dataset = obs["dataset"]
    missing = dataset["missing_counts"]
    actions = []
    if missing.get("age", 0) > 0:
        actions.append({"action_type": "fill_missing", "params": {"column": "age", "strategy": "median"}})
    if missing.get("annual_income", 0) > 0:
        actions.append({"action_type": "fill_missing", "params": {"column": "annual_income", "strategy": "mean"}})
    actions.append({"action_type": "submit", "params": {}})
    return actions


def _rule_actions_task2(obs: Dict) -> List[Dict]:
    """Fix age dtype, remove salary outliers, filter negative purchases, submit."""
    actions = []
    dtypes  = obs["dataset"]["dtypes"]
    # Fix dtype if age is object
    if dtypes.get("age") not in ("int64", "float64", "Int64"):
        actions.append({"action_type": "fix_dtype", "params": {"column": "age", "target_type": "int"}})
    # Remove salary outliers (zscore > 3)
    actions.append({"action_type": "remove_outliers",
                    "params": {"column": "salary", "method": "zscore", "threshold": 3.0}})
    # Filter negative purchase amounts
    actions.append({"action_type": "filter_rows",
                    "params": {"column": "purchase_amount", "operator": "gte", "value": 0}})
    actions.append({"action_type": "submit", "params": {}})
    return actions


def _rule_actions_task3(obs: Dict) -> List[Dict]:
    """Full pipeline: dedup, fill nulls, fix age outliers, standardise city, fix emails, fix dtype."""
    actions = []
    missing = obs["dataset"]["missing_counts"]
    dtypes  = obs["dataset"]["dtypes"]

    # 1. Drop exact duplicates
    actions.append({"action_type": "drop_duplicates", "params": {}})
    # 2. Drop near-dups on customer_id
    actions.append({"action_type": "drop_duplicates", "params": {"subset": ["customer_id"]}})
    # 3. Fill missing age
    if missing.get("age", 0) > 0:
        actions.append({"action_type": "fill_missing",
                        "params": {"column": "age", "strategy": "median"}})
    # 4. Fill missing signup_date
    if missing.get("signup_date", 0) > 0:
        actions.append({"action_type": "fill_missing",
                        "params": {"column": "signup_date", "strategy": "mode"}})
    # 5. Remove age outliers (< 10 or > 100)
    actions.append({"action_type": "filter_rows",
                    "params": {"column": "age", "operator": "gte", "value": 10}})
    actions.append({"action_type": "filter_rows",
                    "params": {"column": "age", "operator": "lte", "value": 100}})
    # 6. Map non-canonical city variants explicitly
    actions.append({"action_type": "map_values", "params": {"column": "city", "mapping": {
        "new york": "New York", "NEW YORK": "New York", "newyork": "New York",
        "los angeles": "Los Angeles", "LOS ANGELES": "Los Angeles", "LA": "Los Angeles",
        "chicago": "Chicago", "CHICAGO": "Chicago",
        "houston": "Houston", "HOUSTON": "Houston",
        "phoenix": "Phoenix", "PHOENIX": "Phoenix",
    }}})
    # 7. Standardise remaining city casing
    actions.append({"action_type": "standardize_text",
                    "params": {"column": "city", "case": "title", "strip": True}})
    # 8. Remove invalid emails (filter rows that don't contain '@' AND '.')
    actions.append({"action_type": "filter_rows",
                    "params": {"column": "email", "operator": "contains", "value": "@"}})
    # 9. Fix purchase_amount dtype (strip '$' and convert)
    if dtypes.get("purchase_amount") not in ("float64", "int64"):
        actions.append({"action_type": "fix_dtype",
                        "params": {"column": "purchase_amount", "target_type": "strip_and_float"}})
    actions.append({"action_type": "submit", "params": {}})
    return actions


RULE_STRATEGIES = {
    "task1_missing_values": _rule_actions_task1,
    "task2_outliers_dtype": _rule_actions_task2,
    "task3_full_pipeline":  _rule_actions_task3,
}


# ── LLM agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a data cleaning agent operating in a structured environment.
You receive an observation in JSON format describing a dataset with quality issues.
Your goal is to fix all issues by choosing actions from the following set:

Actions (respond with a JSON object, no markdown):
  fill_missing      – {"action_type":"fill_missing","params":{"column":"<name>","strategy":"mean|median|mode|value","value":<optional>}}
  remove_outliers   – {"action_type":"remove_outliers","params":{"column":"<name>","method":"zscore|iqr","threshold":<float>}}
  fix_dtype         – {"action_type":"fix_dtype","params":{"column":"<name>","target_type":"int|float|str|datetime|strip_and_float"}}
  drop_duplicates   – {"action_type":"drop_duplicates","params":{"subset":["<col>",...]}}
  normalize         – {"action_type":"normalize","params":{"column":"<name>","method":"minmax|zscore"}}
  filter_rows       – {"action_type":"filter_rows","params":{"column":"<name>","operator":"gt|lt|gte|lte|eq|ne|contains|not_contains","value":<val>}}
  standardize_text  – {"action_type":"standardize_text","params":{"column":"<name>","case":"lower|upper|title","strip":true}}
  submit            – {"action_type":"submit","params":{}}

Rules:
- Output ONLY a valid JSON action object, nothing else.
- Call submit when you believe the dataset is clean.
- Do not over-clean: preserve as many valid rows as possible.
""").strip()


def _llm_action(obs: Dict) -> Dict:
    """Call the LLM and parse the action."""
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    obs_text = json.dumps(obs, indent=2)[:6000]  # truncate for context limit
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Current observation:\n{obs_text}\n\nChoose your next action:"},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Core run function (importable by app.py /baseline endpoint) ───────────────

def run_baseline_task(task_id: str, use_llm: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """
    Run one task episode with the baseline agent.
    Returns dict compatible with BaselineTaskResult.
    """
    # Reset via HTTP if server is running, otherwise use env directly
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        r.raise_for_status()
        obs = r.json()
        use_http = True
    except Exception:
        # Fall back to in-process env
        from env import DataCleaningEnvironment, Action as EnvAction
        _local_env = DataCleaningEnvironment()
        obs_obj    = _local_env.reset(task_id=task_id)
        obs        = obs_obj.model_dump()
        use_http   = False

    strategy    = RULE_STRATEGIES[task_id]
    actions     = strategy(obs)
    final_score = 0.0
    breakdown   = {}
    steps_taken = 0

    for action_dict in actions:
        steps_taken += 1
        if verbose:
            print(f"  [{task_id}] step {steps_taken}: {action_dict['action_type']}")

        if use_http:
            resp = requests.post(f"{BASE_URL}/step", json=action_dict, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        else:
            from env import Action as EnvAction
            result = _local_env.step(EnvAction(**action_dict)).model_dump()

        if result.get("done"):
            final_score = result["reward"]["score"]
            breakdown   = result["info"].get("breakdown", {})
            break

    return {
        "task_id":    task_id,
        "score":      final_score,
        "steps_taken": steps_taken,
        "breakdown":  breakdown,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline agent for Customer Data Cleaning OpenEnv")
    parser.add_argument("--llm",     action="store_true", help="Use LLM agent (requires OPENAI_API_KEY)")
    parser.add_argument("--task",    default=None,        help="Run a single task by ID")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step actions")
    args = parser.parse_args()

    if args.llm and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    tasks = [args.task] if args.task else TASK_IDS
    print(f"\n{'='*60}")
    print(f"  Customer Data Cleaning — Baseline Evaluation")
    print(f"  Agent: {'LLM (' + LLM_MODEL + ')' if args.llm else 'Rule-based'}")
    print(f"{'='*60}\n")

    results = []
    for task_id in tasks:
        print(f"Running {task_id} ...")
        t0 = time.time()
        if args.llm:
            # LLM mode: run interactively
            result = _run_llm_task(task_id, verbose=args.verbose)
        else:
            result = run_baseline_task(task_id, use_llm=False, verbose=args.verbose)
        elapsed = time.time() - t0
        results.append(result)
        print(f"  Score:      {result['score']:.4f}")
        print(f"  Steps:      {result['steps_taken']}")
        print(f"  Breakdown:  {result['breakdown']}")
        print(f"  Time:       {elapsed:.1f}s\n")

    mean = sum(r["score"] for r in results) / len(results)
    print(f"{'='*60}")
    print(f"  Mean score across {len(results)} task(s): {mean:.4f}")
    print(f"{'='*60}\n")

    # Write results to JSON for CI consumption
    out = {
        "agent":      "llm" if args.llm else "rule-based",
        "model":      LLM_MODEL if args.llm else "rule-based-baseline",
        "tasks":      results,
        "mean_score": mean,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Results saved to baseline_results.json")


def _run_llm_task(task_id: str, verbose: bool = False) -> Dict[str, Any]:
    """LLM agent loop."""
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        obs = r.json()
    except Exception:
        from env import DataCleaningEnvironment
        _local_env = DataCleaningEnvironment()
        obs = _local_env.reset(task_id=task_id).model_dump()
        use_http = False
    else:
        use_http = True

    max_steps = TASK_REGISTRY[task_id].get_metadata()["max_steps"]
    score, breakdown, steps_taken = 0.0, {}, 0

    for _ in range(max_steps):
        steps_taken += 1
        try:
            action_dict = _llm_action(obs)
        except Exception as e:
            print(f"    LLM parse error: {e}")
            action_dict = {"action_type": "submit", "params": {}}

        if verbose:
            print(f"  [{task_id}] step {steps_taken}: {action_dict.get('action_type')}")

        if use_http:
            resp   = requests.post(f"{BASE_URL}/step", json=action_dict, timeout=10)
            result = resp.json()
        else:
            from env import Action as EnvAction
            result = _local_env.step(EnvAction(**action_dict)).model_dump()

        obs = result["observation"]
        if result.get("done"):
            score     = result["reward"]["score"]
            breakdown = result["info"].get("breakdown", {})
            break

    return {"task_id": task_id, "score": score, "steps_taken": steps_taken, "breakdown": breakdown}


if __name__ == "__main__":
    main()
