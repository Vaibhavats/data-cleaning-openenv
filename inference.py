from baseline import run_baseline_task
from env.tasks import TASK_REGISTRY

def main():
    results = []

    for task_id in TASK_REGISTRY:
        result = run_baseline_task(task_id)
        results.append(result)

    mean_score = sum(r["score"] for r in results) / len(results)

    print("Model: rule-based")
    print("Results:")
    for r in results:
        print(r)

    print("Mean Score:", round(mean_score, 4))


if __name__ == "__main__":
    main()