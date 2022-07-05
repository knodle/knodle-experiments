from typing import Dict
from torch.optim import AdamW

from knodle.trainer.knn_aggregation.knn import kNNAggregationTrainer
from knodle.trainer.knn_aggregation.config import kNNConfig

from knodle_experiments.training.model import get_model
from knodle_experiments.data.preprocess import create_tfidf_input


def get_knn_config(model, config):
    params = config.get("hyp_params")
    k = params.get("k")
    k = 1 if k is None else k
    custom_model_config = kNNConfig(
        optimizer=AdamW,
        epochs=params.get('num_epochs'),
        k=k, radius=params.get("radius", None),
        filter_non_labelled=params.get("filter_empty_labels")
    )

    return custom_model_config


def load_tfidf_neighborhood_matrix(config: Dict, data_dict: Dict = None):
    if config.get("hyp_params").get("knn_features", None) is None:
        return None

    if data_dict is None:
        raise ValueError("Tfidf for KNN is supposed to be created, but no data dict is given")

    preprocessed_data_dict = create_tfidf_input(
        data_dict=data_dict,
        num_features=config.get("hyp_params").get("num_features")
    )
    return preprocessed_data_dict.get("train_x").tensors[0].numpy()


def get_knn_trainer(
        processed_data_dict: Dict, config: Dict, data_dict: Dict = None
) -> kNNAggregationTrainer:
    """Train a logistic regression model; with SimpleDsModelTrainer."""

    knn_feature_matrix = load_tfidf_neighborhood_matrix(config=config, data_dict=data_dict)

    model = get_model(config, processed_data_dict.get("mapping_rules_labels_t").shape[1])

    custom_model_config = get_knn_config(model, config)

    trainer = kNNAggregationTrainer(
        model=model,
        mapping_rules_labels_t=processed_data_dict.get("mapping_rules_labels_t"),
        model_input_x=processed_data_dict.get("train_x"),
        rule_matches_z=processed_data_dict.get("train_rule_matches_z"),
        dev_model_input_x=processed_data_dict.get("dev_x", None),
        dev_gold_labels_y=processed_data_dict.get("dev_y", None),
        trainer_config=custom_model_config,
        knn_feature_matrix=knn_feature_matrix
    )
    return trainer
