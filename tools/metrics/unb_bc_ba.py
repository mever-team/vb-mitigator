import fairbench as fb

unb_bc_ba_dict = {"best": "high", "performance": "unb"}


def unb_bc_ba(data_dict):
    sensitive_keys = [
        key for key in data_dict.keys() if key not in ["targets", "predictions"]
    ]
    for key in sensitive_keys:
        data_dict[key] = [f"{key}_{value}" for value in data_dict[key]]
    sensitive_keys.append("targets")
    sensitive = fb.Fork(*[fb.categories @ data_dict[key] for key in sensitive_keys])
    sensitive = sensitive.intersectional()

    y = data_dict["targets"]
    yhat = data_dict["predictions"]

    report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
    # fb.describe(report)
    out = {"unb": report.min.accuracy}
    return out


if __name__ == "__main__":

    data_dict = {
        "targets": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "predictions": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        "sensitive_attribute_1": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        "sensitive_attribute_2": [0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    }
    _ = unb_bc_ba(data_dict)
