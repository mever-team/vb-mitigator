from collections import defaultdict

import numpy as np

from .registry import register_metric


@register_metric("wg_ovr", performance="overall", best="high")
def wg_ovr(data):
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    targets = data["targets"]
    predictions = data["predictions"]
    sensitive_attrs = [
        data[key] for key in data if key not in {"targets", "predictions", "ba_groups"}
    ]

    for i in range(len(targets)):
        group_key = (targets[i],) + tuple(attr[i] for attr in sensitive_attrs)
        groups[group_key]["total"] += 1
        if targets[i] == predictions[i]:
            groups[group_key]["correct"] += 1

    accuracies = {
        key: val["correct"] / val["total"]
        for key, val in groups.items()
        if val["total"] > 0
    }
    worst_group_acc = min(accuracies.values(), default=0.0)
    avg_group_acc = sum(accuracies.values()) / len(accuracies) if accuracies else 0.0

    out = {
        "worst_group_accuracy": round(worst_group_acc, 3),
        "overall": round(avg_group_acc, 3),
    }
    return out


if __name__ == "__main__":

    data_dict = {
        "predictions": np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1]),
        "targets": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "background": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
        "object": np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
    }
    _ = wg_ovr(data_dict)
