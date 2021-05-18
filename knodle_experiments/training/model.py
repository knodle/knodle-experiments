from typing import Dict

from transformers import AutoModelForSequenceClassification

from knodle.model.logistic_regression_model import LogisticRegressionModel


def get_model(config: Dict, num_classes: int = 2):
    model_config = config.get("model")
    if model_config.get("type") == "transformer":
        model_type = model_config.get("name")
        model = AutoModelForSequenceClassification.from_pretrained(model_type)
    elif model_config.get("type") == "logistic_regression":
        model = LogisticRegressionModel(config.get("hyp_params").get('num_features'), num_classes)
    else:
        raise ValueError("The only supported model types are transformer and log regs")
    model.train()
    return model
