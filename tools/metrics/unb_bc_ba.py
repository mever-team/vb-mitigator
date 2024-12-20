import numpy as np
import pandas as pd
import fairbench as fb

unb_bc_ba_dict = {"best": "high", "performance": "unb_acc"}


def unb_bc_ba(data_dict):
    sensitive_keys = [
        key for key in data_dict.keys() if key not in ["targets", "predictions", "ba_groups"]
    ]
    #remove the ba_groups from the data_dict
    ba_groups = data_dict.pop('ba_groups')
    # Convert to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data_dict)

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