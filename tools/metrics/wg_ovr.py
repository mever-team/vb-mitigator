import fairbench as fb
import numpy as np

wg_ovr_dict = {"best": "high", "performance": "worst_group_accuracy"}


def wg_ovr(data_dict):
    sensitive_keys = [
        key for key in data_dict.keys() if key not in ["targets", "predictions"]
    ]
    for key in sensitive_keys:
        data_dict[key] = [f"{key}_{value}" for value in data_dict[key]]
    sensitive_keys.append("targets")
    sensitive = fb.Dimensions(*[fb.categories @ data_dict[key] for key in sensitive_keys])
    sensitive = sensitive.intersectional()  # automatically find non-empty intersections
    sensitive = sensitive.strict()  # keep only intersections that have no children
    # print(sensitive.keys().values.values())
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
    report = fb.reports.pairwise(predictions=yhat, labels=y, sensitive=sensitive)
    print(report.acc["sensitive_attribute_1_1&1"])    
   







    out = {
        "worst_group_accuracy": report.min.acc,
        "overall": round(acc, 3),
    }
    # print(out)
    return out


if __name__ == "__main__":

    data_dict = {
        "targets": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        "predictions": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
        "sensitive_attribute_1": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0]),
    }
    _ = wg_ovr(data_dict)
