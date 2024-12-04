from itertools import product

wg_ovr_dict = {"best": "high", "performance": "worst_group_accuracy"}


def wg_ovr(data_dict):
    targets = data_dict["targets"]
    predictions = data_dict["predictions"]
    group_labels = {k: v for k, v in data_dict.items() if k not in ["predictions"]}

    total_correct = sum(t == p for t, p in zip(targets, predictions))
    overall_accuracy = total_correct / len(targets)

    group_accuracies = []
    group_values = list(group_labels.values())

    for combination in product(*group_values):
        group_indices = [
            i for i, combo in enumerate(zip(*group_values)) if combo == combination
        ]
        if group_indices:
            group_targets = [targets[i] for i in group_indices]
            group_predictions = [predictions[i] for i in group_indices]
            group_correct = sum(
                t == p for t, p in zip(group_targets, group_predictions)
            )
            group_accuracy = group_correct / len(group_targets)
            group_accuracies.append(group_accuracy)

    worst_group_accuracy = min(group_accuracies) if group_accuracies else float("inf")

    return {
        "overall_accuracy": overall_accuracy,
        "worst_group_accuracy": worst_group_accuracy,
    }
