import requests

BASE_URL = "https://vaibhavats-data-cleaning-openenv.hf.space"

TASKS = [
    "task1_missing_values",
    "task2_outliers_dtype",
    "task3_full_pipeline"
]

for task in TASKS:

    # START
    print(f"[START] task={task}", flush=True)

    # RESET
    res = requests.post(f"{BASE_URL}/reset", json={"task_id": task})
    obs = res.json()

    step_count = 0

    # VERY SIMPLE LOOP (baseline style)
    for i in range(3):
        step_count += 1

        action = {
            "action": "noop"
        }

        step_res = requests.post(f"{BASE_URL}/step", json=action)
        result = step_res.json()

        reward = result.get("reward", 0)

        print(f"[STEP] step={step_count} reward={reward}", flush=True)

        if result.get("done"):
            break

    # GET FINAL STATE
    state = requests.get(f"{BASE_URL}/state").json()

    # GRADER
    grade = requests.post(
        f"{BASE_URL}/grader",
        json={
            "task_id": task,
            "final_data": state["data"]
        }
    ).json()

    score = grade.get("score", 0)

    # END
    print(f"[END] task={task} score={score} steps={step_count}", flush=True)