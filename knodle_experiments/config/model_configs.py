from typing import Dict
import os


def get_standard_logistic_regression_config(dataset: str) -> Dict:
    config = {
        "project": "knodle",
        "dataset": dataset,
        "data_dir": os.path.join(os.getcwd(), "data", dataset),
        "model": {
            "name": "logistic_regression",
            "type": "logistic_regression"
        },
        "hyp_params": {
            # preprocessing
            "num_features": 3000,
            # training related
            "num_epochs": 8,
            "batch_size": 64,
            "learning_rate": 0.01,
            "filter_non_labelled": True
        }
    }
    return config


def get_standard_bert_config(dataset: str) -> Dict:
    config = {
        "project": "knodle",
        "dataset": dataset,
        "data_dir": os.path.join(os.getcwd(), "data", dataset),
        "model": {
            "type": "transformer",
            "name": "distilbert-base-uncased"
        },
        "hyp_params": {
            # training related
            "num_epochs": 2,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "filter_non_labelled": True
        }
    }
    return config
