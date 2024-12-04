acc_dict = {"best": "high", "performance": "accuracy"}


def acc(data_dict):
    predictions = data_dict["predictions"]
    targets = data_dict["targets"]

    accuracy = (predictions == targets).mean()
    return {"accuracy": accuracy}
