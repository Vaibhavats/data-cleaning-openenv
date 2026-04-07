# 🧹 Customer Data Cleaning — OpenEnv

> An RL environment where AI agents learn to clean messy real-world customer datasets.  
> Built to the **OpenEnv** specification with 3 graded tasks (easy → hard), dense rewards, and a reproducible baseline.
---
title: Data Cleaning OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

---

## 🎯 Why This Environment?

Data cleaning is one of the most time-consuming tasks in data engineering and analytics — studies consistently show it consumes 60–80 % of a data professional's time. Every production ML pipeline begins with it. Yet no existing OpenEnv benchmark targets this domain.

This environment teaches agents to:
- **detect** data quality issues (nulls, outliers, type errors, duplicates, format inconsistencies)
- **apply** the correct cleaning strategy for each issue
- **sequence** operations in the right order without destroying valid data

Agents that score well here transfer directly to real ETL and data-prep workflows.

---

## 🗂️ Project Structure

```
dataclean-env/
├── app.py                        # FastAPI server (all endpoints)
├── baseline.py                   # Baseline inference script (rule-based + LLM)
├── openenv.yaml                  # OpenEnv metadata
├── Dockerfile                    # Container definition
├── requirements.txt
├── README.md
└── env/
    ├── __init__.py
    ├── models.py                 # Typed Pydantic models (Observation, Action, Reward…)
    ├── environment.py            # DataCleaningEnvironment: reset/step/state
    └── tasks/
        ├── __init__.py
        ├── task1_missing_values.py    # Easy task
        ├── task2_outliers_dtype.py    # Medium task
        └── task3_full_pipeline.py     # Hard task
```

---

## 🧩 Observation Space

Every call to `reset()` and `step()` returns an **Observation** with:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Active task identifier |
| `task_description` | `str` | Natural-language description of what to clean |
| `step_count` | `int` | Steps taken this episode |
| `dataset.columns` | `list[str]` | Column names |
| `dataset.dtypes` | `dict[str,str]` | Per-column dtype (e.g. `"object"`, `"float64"`) |
| `dataset.shape` | `[rows, cols]` | Current dataset dimensions |
| `dataset.data` | `list[dict]` | Up to 200 rows as JSON records |
| `dataset.missing_counts` | `dict[str,int]` | Null count per column |
| `dataset.stats` | `dict` | Per-column stats (mean, std, min, max, unique, top values) |
| `action_history` | `list[dict]` | Last 10 actions + their effects |
| `reward_so_far` | `float` | Cumulative reward this episode [0, 1] |
| `hints` | `list[str]?` | Optional hints after bad actions |

---

## ⚡ Action Space

All actions are JSON objects with `action_type` and `params`:

| `action_type` | Required `params` | Effect |
|---|---|---|
| `fill_missing` | `column`, `strategy` (mean/median/mode/value), `value`? | Impute nulls |
| `remove_outliers` | `column`, `method` (zscore/iqr), `threshold` | Drop outlier rows |
| `fix_dtype` | `column`, `target_type` (int/float/str/datetime/strip_and_float) | Cast column type |
| `drop_duplicates` | `subset`? (list of columns) | Remove duplicate rows |
| `normalize` | `column`, `method` (minmax/zscore) | Normalise numeric column |
| `filter_rows` | `column`, `operator` (gt/lt/gte/lte/eq/ne/contains/not_contains), `value` | Keep matching rows |
| `standardize_text` | `column`, `case` (lower/upper/title), `strip` (bool) | Normalise string case |
| `submit` | *(empty)* | End episode, trigger final grader |

**Example action:**
```json
{"action_type": "fill_missing", "params": {"column": "age", "strategy": "median"}}
```

---

## 🏆 Tasks

### Task 1 — Fill Missing Values *(Easy)*

**Dataset:** 40-row customer table with `customer_id`, `name`, `email`, `age`, `annual_income`, `city`.

**Issues:** 6 missing `age` values (correct fix: median imputation) + 5 missing `annual_income` values (correct fix: mean imputation).

**Max steps:** 15

**Grader dimensions (weights):**
| Dimension | Weight | Criterion |
|---|---|---|
| Completeness | 40 % | All nulls filled |
| Accuracy | 40 % | Correct imputation values used |
| Preservation | 20 % | All 40 rows retained |

**Expected baseline score:** ~0.97

---

### Task 2 — Fix Outliers and Data Types *(Medium)*

**Dataset:** 60-row sales table with `sale_id`, `rep_name`, `age`, `salary`, `purchase_amount`, `region`.

**Issues:**
- `age` stored as `object` strings → cast to numeric
- `salary` has 5 extreme outliers (> 3σ above mean)
- `purchase_amount` has 4 negative values (invalid)

**Max steps:** 20

**Grader dimensions (weights):**
| Dimension | Weight | Criterion |
|---|---|---|
| dtype_fix | 25 % | `age` is numeric dtype |
| outlier_removal | 35 % | F1 score on known outlier rows |
| negative_purchase | 25 % | No negative purchase_amount |
| preservation | 15 % | ≥ 51 valid rows kept |

**Expected baseline score:** ~0.85

---

### Task 3 — Full Data Cleaning Pipeline *(Hard)*

**Dataset:** 100-row CRM export with `customer_id`, `full_name`, `email`, `age`, `city`, `purchase_amount`, `signup_date`.

**Issues (7 types compounded):**
1. 7 exact duplicate rows
2. 8 near-duplicate rows (same `customer_id`, different `full_name`)
3. 6 missing `age` values
4. 4 missing `signup_date` values
5. 5 unrealistic ages (< 10 or > 100)
6. Inconsistent city casing (`"new york"`, `"NYC"`, etc.)
7. 5 invalid email addresses
8. `purchase_amount` stored as `"$123.45"` strings

**Max steps:** 30

**Grader dimensions (weights):**
| Dimension | Weight | Criterion |
|---|---|---|
| deduplication | 20 % | Exact + customer_id dups removed |
| missing_filled | 15 % | age + signup_date nulls filled |
| age_validity | 15 % | No ages < 10 or > 100 |
| city_standard | 15 % | All cities in canonical Title Case set |
| email_validity | 20 % | No invalid email rows remain |
| dtype_purchase | 15 % | `purchase_amount` is float64 |

**Expected baseline score:** ~0.72

---

## 🎁 Reward Function

Rewards are **dense** — the agent receives a signal after *every* action, not just at the end.

**Step reward components:**
- `+credit` for each measurable improvement (nulls filled, outliers removed, dtypes fixed, etc.)
- `-penalty` for over-deletion of valid rows (0.03–0.05 per extra row lost)
- `-0.02` for invalid/malformed actions

**Final (submit) reward:**
- Weighted combination of grader sub-scores (0.0–1.0)
- Deterministic and reproducible — same dataset, same actions → same score

This design means:
- An agent that does *nothing* scores 0.0
- An agent that destroys the dataset scores near 0.0
- An agent that makes partial progress scores proportionally
- A perfect cleaner scores 1.0

---

## 🚀 Setup & Usage

### Option 1: Docker (recommended)

```bash
git clone https://github.com/<your-handle>/dataclean-env
cd dataclean-env
docker build -t dataclean-env .
docker run -p 7860:7860 dataclean-env
```

### Option 2: Local Python

```bash
pip install -r requirements.txt
python app.py
```

### Run the baseline

```bash
# Rule-based (no API key needed)
python baseline.py

# LLM agent
OPENAI_API_KEY=sk-... python baseline.py --llm --verbose
```

---

## 🔌 API Reference

All endpoints return JSON.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | Start episode: `{"task_id": "task1_missing_values"}` |
| `POST` | `/step` | Apply action: `{"action_type": "...", "params": {...}}` |
| `GET` | `/state` | Raw environment state (debugging) |
| `GET` | `/tasks` | List all tasks + action schema |
| `POST` | `/grader` | Standalone grader (submit final dataset) |
| `POST` | `/baseline` | Trigger built-in baseline on all 3 tasks |

### Quick example

```python
import requests

BASE = "http://localhost:7860"

# Start Task 1
obs = requests.post(f"{BASE}/reset", json={"task_id": "task1_missing_values"}).json()
print("Missing counts:", obs["dataset"]["missing_counts"])

# Fill age with median
result = requests.post(f"{BASE}/step", json={
    "action_type": "fill_missing",
    "params": {"column": "age", "strategy": "median"}
}).json()
print("Step reward:", result["reward"]["score"])

# Submit
result = requests.post(f"{BASE}/step", json={"action_type": "submit", "params": {}}).json()
print("Final score:", result["reward"]["score"])
print("Breakdown:", result["info"]["breakdown"])
```

---

## 📊 Baseline Scores

Measured with the deterministic rule-based agent (`python baseline.py`):

| Task | Difficulty | Score | Steps | Notes |
|---|---|---|---|---|
| task1_missing_values | Easy | **1.00** | 3 | Perfect — median/mean imputation correct |
| task2_outliers_dtype | Medium | **0.79** | 4 | Single-pass z-score catches 2/5 outliers; iterative or IQR gets higher |
| task3_full_pipeline | Hard | **0.85** | 10 | Requires all 8 cleaning steps in correct order |
| **Mean** | | **0.88** | | |

*Scores are fully reproducible — fixed random seeds (`np.random.seed(42/7/99)`) in all dataset generators.*

### Why Task 2 isn't 1.0 for the rule-based baseline

A naïve single-pass z-score computed on the **full** dataset (including the 5 extreme outliers) raises the mean and std, making only 2 of the 5 outliers cross the 3σ threshold. An optimal agent must either:
- Apply z-score **iteratively** until no new outliers are found
- Use **IQR** method which is more robust to extreme values  
- Apply a direct `filter_rows` action with an empirically chosen threshold

This creates a meaningful gap between the baseline and an optimal policy — exactly the kind of signal needed to train and evaluate agents.

---

## 🤖 Agent Development Tips

1. **Always inspect `dataset.missing_counts` and `dataset.dtypes` first** — they surface the most obvious issues.
2. **Read `action_history`** — past rewards tell you what worked.
3. **Don't over-clean** — row-deletion penalties accumulate; only remove rows with confirmed issues.
4. **Order matters for Task 3**: dedup → fill nulls → filter invalids → fix dtypes → submit.
5. **`submit` early if stuck** — a partial score is better than hitting max_steps with no submit (which also triggers grading).

---

## 📜 License

MIT
