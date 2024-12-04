from models import models_dict


def get_model(model_name, num_class):
    model = models_dict[model_name](num_class)
    return model
