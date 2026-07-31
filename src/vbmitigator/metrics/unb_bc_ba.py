import numpy as np
import pandas as pd

from .registry import register_metric


@register_metric("unb_bc_ba", performance="unb_acc", best="high")
def unb_bc_ba(data_dict):
    sensitive_keys = [
        key for key in data_dict.keys() if key not in ["targets", "predictions", "ba_groups"]
    ]
    # Read (without mutating the caller's dict) the bias-aligned group list.
    ba_groups = data_dict.get("ba_groups", [])
    # Convert to a DataFrame from the per-sample arrays only (exclude ba_groups,
    # whose length differs from the number of samples).
    df = pd.DataFrame(
        {k: v for k, v in data_dict.items() if k != "ba_groups"}
    )

    # Create a new column for the subgroup defined by (target, sensitive_attribute_1)
    df['subgroup'] = list(zip(df['targets'], df[sensitive_keys[0]]))

    # Initialize list to hold subgroup metrics
    subgroup_metrics = []

    # Calculate accuracy and counts for each subgroup
    for subgroup in df['subgroup'].unique():
        subgroup_data = df[df['subgroup'] == subgroup]
        targets = subgroup_data['targets'].values
        predictions = subgroup_data['predictions'].values

        # Calculate accuracy for the subgroup
        accuracy = np.mean(predictions == targets)

        # Store metrics for the subgroup
        subgroup_metrics.append({
            'subgroup': subgroup,
            'accuracy': accuracy,
            'count': len(subgroup_data)
        })
    # Convert metrics list to DataFrame for easier analysis
    metrics_df = pd.DataFrame(subgroup_metrics)

    # Classify groups based on the threshold
    metrics_df['is_ba'] = metrics_df['subgroup'].isin(ba_groups)

    # Calculate average accuracies for BA and BC groups
    ba_accuracy = metrics_df[metrics_df['is_ba']]['accuracy'].mean() if not metrics_df[metrics_df['is_ba']].empty else 0
    bc_accuracy = metrics_df[~metrics_df['is_ba']]['accuracy'].mean() if not metrics_df[~metrics_df['is_ba']].empty else 0

    # Calculate overall average accuracy across all groups
    overall_accuracy = metrics_df['accuracy'].mean()

    # Display results
    out = {
    "unb_acc": overall_accuracy,
    "ba_acc": ba_accuracy,
    "bc_acc": bc_accuracy,
    # "detailed": metrics_df.to_dict(orient='records')
    }
    # print(metrics_df)
    return out


if __name__ == "__main__":

    data_dict = {
        "targets": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "predictions": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        "sensitive_attribute_1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        "ba_groups": [(0, 0), (1, 1)]
    }
    out = unb_bc_ba(data_dict)
    print(out)

# 00 -> 0.5
# 01 -> 0
# 10 -> 0
# 11 -> 1
