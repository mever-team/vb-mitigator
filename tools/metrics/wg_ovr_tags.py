import numpy as np

wg_ovr_tags_dict = {"best": "high", "performance": "overall_accuracy"}


def wg_ovr_tags(data_dict):
    predictions = np.array(data_dict["predictions"])
    targets = np.array(data_dict["targets"])

    # Identify class-tag keys (excluding targets and predictions)
    class_tag_keys = [
        key for key in data_dict.keys() if key not in ["targets", "predictions"]
    ]

    # Identify unique classes
    classes = set(key.split("_")[0] for key in class_tag_keys)

    # Initialize storage for group accuracies
    group_accuracies = {}

    # Overall accuracy
    overall_accuracy = (predictions == targets).mean() * 100

    for class_name in classes:
        # Create masks for samples of the given class
        class_mask = np.any(
            [
                np.array(data_dict[key]) == 1
                for key in class_tag_keys
                if key.startswith(class_name)
            ],
            axis=0,
        )
        # print(f"{class_name}, mask: {class_mask}")
        class_idx = targets[class_mask][0]

        # Samples with at least one tag
        correct_with_tag = (
            (predictions[class_mask] == targets[class_mask]).sum()
            if class_mask.sum() > 0
            else 0
        )
        total_samples_with_tag = class_mask.sum()
        acc_with_tag = (
            (correct_with_tag / total_samples_with_tag * 100)
            if total_samples_with_tag > 0
            else 0
        )

        lst = [
            np.array(data_dict[key]) == 0
            for key in class_tag_keys
            if key.startswith(class_name)
        ]
        lst.append(targets == class_idx)
        no_tag_mask = np.all(
            lst,
            axis=0,
        )
        # no_tag_mask = ~class_mask
        # print(f"{class_name}, class_idx: {class_idx}, ~mask: {no_tag_mask}")
        correct_without_tag = (
            (predictions[no_tag_mask] == targets[no_tag_mask]).sum()
            if no_tag_mask.sum() > 0
            else 0
        )
        total_samples_without_tag = no_tag_mask.sum()
        acc_without_tag = (
            (correct_without_tag / total_samples_without_tag * 100)
            if total_samples_without_tag > 0
            else 0
        )

        # Store accuracies in separate keys
        group_accuracies[f"{class_name}_has_tag"] = acc_with_tag
        group_accuracies[f"{class_name}_no_tag"] = acc_without_tag

    # Compute worst group accuracy and corresponding group key
    worst_group_acc = min(group_accuracies.values())
    worst_group_key = min(group_accuracies, key=group_accuracies.get)

    return {
        "overall_accuracy": overall_accuracy,
        **group_accuracies,
        "worst_group_accuracy": worst_group_acc,
        "worst_group": worst_group_key,
    }


def main():
    # Example data for testing
    data_dict = {
        "predictions": [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        "targets": [0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        "class1_tagA0": [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "class1_tagA1": [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
        "class2_tagB0": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "class2_tagB1": [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    }
    # class1 has tags: 0,1,2,5,9 acc: 4/5,
    # class1 no tags: 8, 1/1
    # class2 has tags: 3,4,6 acc: 3/3
    # class2 no tags: 7,9, acc: 1/2

    results = wg_ovr_tags(data_dict)

    # Print the results
    print("Results:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
