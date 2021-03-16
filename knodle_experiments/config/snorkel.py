from typing import Dict

from knodle_experiments.config.model_configs import (
    get_standard_bert_config, get_standard_logistic_regression_config
)


def get_logistic_regression_config(dataset: str) -> Dict:
    config = get_standard_logistic_regression_config(dataset)
    config["method"] = "snorkel"
    return config


def get_knn_logistic_regression_config(dataset: str) -> Dict:
    config = get_standard_logistic_regression_config(dataset)
    config["method"] = "snorkel"
    config["hyp_params"]["k"] = 2
    return config


def get_bert_config(dataset: str) -> Dict:
    config = get_standard_bert_config(dataset)
    config["method"] = "snorkel"
    return config


def get_knn_bert_config(dataset: str) -> Dict:
    config = get_standard_bert_config(dataset)
    config["method"] = "snorkel"
    config["hyp_params"]["k"] = 2
    config["hyp_params"]["knn_features"] = "tfidf"
    config["hyp_params"]["num_features"] = 3000
    return config
