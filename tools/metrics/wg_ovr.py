import fairbench as fb
import numpy as np

wg_ovr_dict = {"best": "high", "performance": "worst_group_accuracy"}


def wg_ovr(data_dict):
    print(data_dict)
    sensitive_keys = [
        key for key in data_dict.keys() if key not in ["targets", "predictions"]
    ]
    for key in sensitive_keys:
        data_dict[key] = [f"{key}_{value}" for value in data_dict[key]]
    sensitive_keys.append("targets")
    sensitive = fb.Fork(*[fb.categories @ data_dict[key] for key in sensitive_keys])
    sensitive = sensitive.intersectional()
    print(sensitive)

    y = data_dict["targets"]
    yhat = data_dict["predictions"]

    # workarround for multiple classes scenarios
    yhat = data_dict["predictions"] == data_dict["targets"]
    yhat = yhat.astype(float)
    acc = np.sum(yhat) / yhat.size
    y = np.zeros_like(y) + 1
    # reports = {}
    # for cl in set(y):
    #     cly = y == cl
    #     clyhat = yhat == cl
    #     reports[str(cl)] = fb.multireport(
    #         predictions=clyhat, labels=cly, sensitive=sensitive
    #     )
    # all_reports = fb.Fork(reports)
    report = fb.unireport(
        predictions=yhat, labels=y, sensitive=sensitive, metrics=[fb.accuracy]
    )
    # print(report.min.accuracy)
    # # print(np.mean(list(all_reports.min.accuracy.branches().values())))

    # fb.text_visualize(report)
    out = {
        "worst_group_accuracy": report.min.accuracy,
        "overall": round(acc, 3),
    }
    # print(out)
    return out


if __name__ == "__main__":

    data_dict = {
        "targets": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "predictions": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        "sensitive_attribute_1": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        # "sensitive_attribute_2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    _ = wg_ovr(data_dict)
