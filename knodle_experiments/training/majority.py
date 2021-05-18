from typing import Dict

from transformers import AdamW

from knodle.trainer.baseline.config import MajorityConfig
from knodle.trainer.baseline.majority import MajorityVoteTrainer

from knodle_experiments.training.model import get_model


def get_majority_trainer_config(model, config):
    custom_model_config =MajorityConfig(
        optimizer=AdamW,
        batch_size=config.get('hyp_params').get('batch_size'),
        epochs=config.get('hyp_params').get('num_epochs'),
        filter_non_labelled=config.get("hyp_params").get("filter_non_labelled")
    )

    return custom_model_config


def get_majority_trainer(
        processed_data_dict: Dict, config: Dict, data_dict: Dict = None
) -> MajorityVoteTrainer:
    """Train a logistic regression model; with SimpleDsModelTrainer."""

    model = get_model(config, processed_data_dict.get("mapping_rules_labels_t").shape[1])

    custom_model_config = get_majority_trainer_config(model, config)

    trainer = MajorityVoteTrainer(
        model,
        mapping_rules_labels_t=processed_data_dict.get("mapping_rules_labels_t"),
        model_input_x=processed_data_dict.get("train_x"),
        rule_matches_z=processed_data_dict.get("train_rule_matches_z"),
        dev_model_input_x=processed_data_dict.get("dev_x", None),
        dev_gold_labels_y=processed_data_dict.get("dev_y", None),
        trainer_config=custom_model_config,
    )
    return trainer
