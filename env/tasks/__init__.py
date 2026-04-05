from . import task1_missing_values, task2_outliers_dtype, task3_full_pipeline

TASK_REGISTRY = {
    "task1_missing_values": task1_missing_values,
    "task2_outliers_dtype": task2_outliers_dtype,
    "task3_full_pipeline":  task3_full_pipeline,
}
