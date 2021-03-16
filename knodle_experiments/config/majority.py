from typing import Dict

from knodle_experiments.config.model_configs import (
    get_standard_bert_config, get_standard_logistic_regression_config
)


def get_logistic_regression_config(dataset: str) -> Dict:
    config = get_standard_logistic_regression_config(dataset)
    config["method"] = "majority"

    return config


def get_bert_config(dataset: str) -> Dict:
    config = get_standard_bert_config(dataset)
    config["method"] = "majority"
    return config