from typing import Dict

from knodle_experiments.training.majority import get_majority_trainer
from knodle_experiments.training.knn import get_knn_trainer
from knodle_experiments.training.snorkel import get_snorkel_trainer


def get_trainer(config: Dict, processed_dict: Dict, data_dict: Dict = None):
    trainer_creation_input = [
        processed_dict.get('train_x'), processed_dict.get('train_rule_matches_z'),
        processed_dict.get('mapping_rules_labels_t'), config
    ]

    if config.get("method") == "majority":
        trainer = get_majority_trainer(processed_data_dict=processed_dict, config=config, data_dict=data_dict)
    elif config.get("method") == "knn":
        trainer = get_knn_trainer(processed_data_dict=processed_dict, config=config, data_dict=data_dict)
    elif config.get("method") == "snorkel":
        trainer = get_snorkel_trainer(processed_data_dict=processed_dict, config=config, data_dict=data_dict)
    else:
        raise ValueError("The given method is not specified")

    trainer.trainer_config.class_weights = None
    trainer.trainer_config.seed = 42
    trainer.trainer_config.output_classes = processed_dict["mapping_rules_labels_t"].shape[1]

    return trainer