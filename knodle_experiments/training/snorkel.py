from typing import Dict

from transformers import AdamW

from knodle.trainer.snorkel.config import SnorkelConfig, SnorkelKNNConfig
from knodle.trainer.snorkel.snorkel import SnorkelTrainer, SnorkelKNNDenoisingTrainer

from knodle_experiments.training.model import get_model

from knodle_experiments.training.knn import load_tfidf_neighborhood_matrix


def get_snorkel_config(model, config):
    params = config.get("hyp_params")
    if params.get("k") is not None:
        custom_model_config = SnorkelKNNConfig(
            optimizer=AdamW,
            epochs=params.get('num_epochs'),
            k=params.get("k", None),
            radius=params.get("radius", None),
            filter_non_labelled=params.get("filter_empty_labels"),
            label_model_num_epochs=params.get("label_model_num_epochs", 5000)
        )
    else:
        custom_model_config = SnorkelConfig(
            optimizer=AdamW(model.parameters(), lr=config.get('hyp_params').get('learning_rate')),
            batch_size=config.get('hyp_params').get('batch_size'),
            epochs=config.get('hyp_params').get('num_epochs'),
            filter_non_labelled=config.get("hyp_params").get("filter_empty_labels"),
            label_model_num_epochs=params.get("label_model_num_epochs", 5000)
        )

    return custom_model_config


def get_snorkel_trainer(
        processed_data_dict: Dict, config: Dict, data_dict: Dict = None
) -> SnorkelTrainer:
    """Train a logistic regression model; with SimpleDsModelTrainer."""

    model = get_model(config, processed_data_dict.get("mapping_rules_labels_t").shape[1])
    custom_model_config = get_snorkel_config(model, config)

    if config.get("hyp_params").get("k") is not None:
        knn_feature_matrix = load_tfidf_neighborhood_matrix(config=config, data_dict=data_dict)

        trainer = SnorkelKNNDenoisingTrainer(
            model=model,
            mapping_rules_labels_t=processed_data_dict.get("mapping_rules_labels_t"),
            model_input_x=processed_data_dict.get("train_x"),
            rule_matches_z=processed_data_dict.get("train_rule_matches_z"),
            dev_model_input_x=processed_data_dict.get("dev_x", None),
            dev_gold_labels_y=processed_data_dict.get("dev_y", None),
            knn_feature_matrix=knn_feature_matrix,
            trainer_config=custom_model_config,
        )
    else:
        trainer = SnorkelTrainer(
            model=model,
            mapping_rules_labels_t=processed_data_dict.get("mapping_rules_labels_t"),
            model_input_x=processed_data_dict.get("train_x"),
            rule_matches_z=processed_data_dict.get("train_rule_matches_z"),
            dev_model_input_x=processed_data_dict.get("dev_x", None),
            dev_gold_labels_y=processed_data_dict.get("dev_y", None),
            trainer_config=custom_model_config
        )
    return trainer
