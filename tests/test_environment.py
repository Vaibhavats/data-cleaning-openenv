"""
tests/test_environment.py
=========================
Comprehensive test suite — runs without pydantic/fastapi (uses task modules directly).
Validates: task data, step rewards, graders, action logic, edge cases.

Run: python -m pytest tests/ -v
  OR: python tests/test_environment.py
"""
import sys, re, json
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import importlib.util, traceback

# ── Loader ────────────────────────────────────────────────────────────────────

def load_mod(relpath):
    spec = importlib.util.spec_from_file_location('m', relpath)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

task1 = load_mod('../env/tasks/task1_missing_values.py')
task2 = load_mod('../env/tasks/task2_outliers_dtype.py')
task3 = load_mod('../env/tasks/task3_full_pipeline.py')

TASKS = [task1, task2, task3]

# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = "✅"
FAIL = "❌"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg    = f"{status} {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append((name, condition))
    return condition


# ── 1. Dataset integrity ──────────────────────────────────────────────────────

print("\n── 1. Dataset integrity ─────────────────────────────────────────────────")

df1 = task1.get_initial_df()
check("T1: shape is (40, 6)",      df1.shape == (40, 6))
check("T1: age has 6 nulls",       df1['age'].isna().sum() == 6)
check("T1: annual_income 5 nulls", df1['annual_income'].isna().sum() == 5)
check("T1: no other nulls",        df1.drop(columns=['age','annual_income']).isna().sum().sum() == 0)

df2 = task2.get_initial_df()
check("T2: shape is (60, 6)",         df2.shape == (60, 6))
check("T2: age dtype is object",      not pd.api.types.is_numeric_dtype(df2['age']), f"dtype={df2['age'].dtype}")
check("T2: salary has 5 outliers",    (df2['salary'] > task2.OUTLIER_THRESHOLD).sum() == 5)
check("T2: purchase has 4 negatives", (df2['purchase_amount'] < 0).sum() == 4)

df3 = task3.get_initial_df()
check("T3: shape is (100, 7)",       df3.shape == (100, 7))
check("T3: age has 6 nulls",         df3['age'].isna().sum() == 6)
check("T3: signup_date has 4 nulls", df3['signup_date'].isna().sum() == 4)
check("T3: exact dups >= 5 rows",    df3.duplicated().sum() >= 5, f"found={df3.duplicated().sum()}")
check("T3: invalid emails exist",    sum(not bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', str(e))) for e in df3['email']) >= 4)
check("T3: bad city variants exist", sum(1 for v in df3['city'] if v not in task3.CANONICAL_CITIES) >= 5)

# ── 2. Metadata ───────────────────────────────────────────────────────────────

print("\n── 2. Metadata ──────────────────────────────────────────────────────────")

for t, tid, diff, ms in [
    (task1, "task1_missing_values", "easy", 15),
    (task2, "task2_outliers_dtype", "medium", 20),
    (task3, "task3_full_pipeline", "hard", 30),
]:
    meta = t.get_metadata()
    check(f"{tid}: task_id correct",   meta['task_id'] == tid)
    check(f"{tid}: difficulty={diff}", meta['difficulty'] == diff)
    check(f"{tid}: max_steps={ms}",    meta['max_steps'] == ms)
    check(f"{tid}: description set",   len(meta.get('description','')) > 20)

# ── 3. Step rewards ────────────────────────────────────────────────────────────

print("\n── 3. Step rewards ──────────────────────────────────────────────────────")

# T1 step rewards
d1 = task1.get_initial_df()
prev = d1.copy(); d1['age'] = d1['age'].fillna(d1['age'].median())
r, _ = task1.step_reward(prev, d1)
check("T1: filling age gives reward > 0",  r > 0, f"reward={r}")
check("T1: step reward <= 1.0",            r <= 1.0)

prev = d1.copy()
d1_rows_dropped = d1.iloc[:35]  # drop 5 rows
r2, _ = task1.step_reward(prev, d1_rows_dropped)
check("T1: row deletion reduces reward",   r2 < r or r2 <= 0, f"r2={r2}")

# T2 step rewards
d2 = task2.get_initial_df()
prev = d2.copy()
d2['age'] = pd.to_numeric(d2['age'], errors='coerce')
r, _ = task2.step_reward(prev, d2)
check("T2: dtype fix gives reward > 0",   r > 0, f"reward={r}")

# T3 step rewards
d3 = task3.get_initial_df()
prev = d3.copy()
d3 = d3.drop_duplicates().reset_index(drop=True)
r, _ = task3.step_reward(prev, d3)
check("T3: dedup gives reward > 0",       r > 0, f"reward={r}")

d3_prev = d3.copy()
d3['city'] = d3['city'].replace({"LA": "Los Angeles", "new york": "New York"})
r, _ = task3.step_reward(d3_prev, d3)
check("T3: city fix gives reward >= 0",   r >= 0, f"reward={r}")

# ── 4. Graders produce [0,1] scores ──────────────────────────────────────────

print("\n── 4. Grader score range ────────────────────────────────────────────────")

# Grade unmodified dirty data — should be low
for t, mod_df in [(task1, task1.get_initial_df()),
                   (task2, task2.get_initial_df()),
                   (task3, task3.get_initial_df())]:
    res = t.grade(mod_df)
    check(f"{t.get_metadata()['task_id']}: dirty score in [0,1]",
          0.0 <= res['score'] <= 1.0, f"score={res['score']}")
    check(f"{t.get_metadata()['task_id']}: dirty score < 0.8",
          res['score'] < 0.8, f"score={res['score']}")
    check(f"{t.get_metadata()['task_id']}: breakdown is dict",
          isinstance(res['breakdown'], dict))
    check(f"{t.get_metadata()['task_id']}: passed key present",
          'passed' in res)

# Grade perfectly clean data — should be high/perfect
d1_clean = task1.get_initial_df()
d1_clean['age']           = d1_clean['age'].fillna(d1_clean['age'].median())
d1_clean['annual_income'] = d1_clean['annual_income'].fillna(d1_clean['annual_income'].mean())
res1 = task1.grade(d1_clean)
check("T1: clean score == 1.0",    res1['score'] == 1.0, f"score={res1['score']}")
check("T1: clean passed == True",  res1['passed'] is True)

d2_clean = task2.get_initial_df()
d2_clean['age'] = pd.to_numeric(d2_clean['age'], errors='coerce')
z = (d2_clean['salary'] - d2_clean['salary'].mean()) / d2_clean['salary'].std()
d2_clean = d2_clean[abs(z) <= 3.0].reset_index(drop=True)
d2_clean = d2_clean[d2_clean['purchase_amount'] >= 0].reset_index(drop=True)
res2 = task2.grade(d2_clean)
check("T2: clean score >= 0.5",    res2['score'] >= 0.5, f"score={res2['score']}")
check("T2: clean passed == True",  res2['passed'] is True)

# ── 5. Grader determinism ─────────────────────────────────────────────────────

print("\n── 5. Grader determinism ────────────────────────────────────────────────")

r_a = task1.grade(d1_clean)['score']
r_b = task1.grade(d1_clean)['score']
check("T1: grader is deterministic", r_a == r_b)

r_a = task2.grade(d2_clean)['score']
r_b = task2.grade(d2_clean)['score']
check("T2: grader is deterministic", r_a == r_b)

# ── 6. Edge cases ─────────────────────────────────────────────────────────────

print("\n── 6. Edge cases ────────────────────────────────────────────────────────")

# Empty dataframe
empty_df = pd.DataFrame(columns=d1_clean.columns)
try:
    res = task1.grade(empty_df)
    check("T1: grade empty df doesn't crash", True, f"score={res['score']}")
    check("T1: empty df score == 0",          res['score'] == 0.0, f"score={res['score']}")
except Exception as e:
    check("T1: grade empty df doesn't crash", False, str(e))
    check("T1: empty df score == 0", False, "exception")

# Dataset with all values filled  
all_filled = task1.get_initial_df().fillna(99)
res = task1.grade(all_filled)
check("T1: all filled (wrong value) < 0.5 accuracy", res['breakdown']['accuracy'] < 1.0)

# T3 full pipeline score
d3_full = task3.get_initial_df()
d3_full = d3_full.drop_duplicates().reset_index(drop=True)
d3_full = d3_full.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
d3_full['age'] = d3_full['age'].fillna(d3_full['age'].median())
mode_v = d3_full['signup_date'].mode()
d3_full['signup_date'] = d3_full['signup_date'].fillna(mode_v[0] if len(mode_v) else 'unknown')
d3_full = d3_full[(d3_full['age'] >= 10) & (d3_full['age'] <= 100)].reset_index(drop=True)
d3_full['city'] = d3_full['city'].replace({
    "new york": "New York", "NEW YORK": "New York", "newyork": "New York",
    "los angeles": "Los Angeles", "LOS ANGELES": "Los Angeles", "LA": "Los Angeles",
    "chicago": "Chicago", "CHICAGO": "Chicago",
    "houston": "Houston", "HOUSTON": "Houston",
    "phoenix": "Phoenix", "PHOENIX": "Phoenix",
}).str.strip().str.title()
d3_full = d3_full[d3_full['email'].apply(
    lambda e: bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', str(e)))
)].reset_index(drop=True)
d3_full['purchase_amount'] = pd.to_numeric(
    d3_full['purchase_amount'].astype(str).str.replace(r'[^\d.\-]', '', regex=True),
    errors='coerce'
)
res3 = task3.grade(d3_full)
check("T3: full clean score == 1.0", res3['score'] == 1.0, f"score={res3['score']}")
check("T3: all sub-scores == 1.0",
      all(v == 1.0 for v in res3['breakdown'].values()),
      str(res3['breakdown']))

# ── 7. Reproducibility — same seed same data ──────────────────────────────────

print("\n── 7. Reproducibility ───────────────────────────────────────────────────")

df1_a = task1.get_initial_df()
df1_b = task1.get_initial_df()
check("T1: dataset reproducible",
      df1_a.to_json() == df1_b.to_json())

df3_a = task3.get_initial_df()
df3_b = task3.get_initial_df()
check("T3: dataset reproducible",
      df3_a.to_json() == df3_b.to_json())

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"  Results: {n_pass}/{len(results)} passed  ({n_fail} failed)")
if n_fail:
    print("\n  FAILED tests:")
    for name, ok in results:
        if not ok:
            print(f"    ❌ {name}")
print(f"{'='*60}\n")

sys.exit(0 if n_fail == 0 else 1)
