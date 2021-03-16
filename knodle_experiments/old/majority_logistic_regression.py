from typing import Dict
from torch.optim import AdamW

from knodle.trainer.baseline.baseline import NoDenoisingTrainer
from knodle.model.logistic_regression.logistic_regression_model import LogisticRegressionModel
from knodle.trainer.baseline.majority_config import MajorityConfig


def get_majority_logistic_regression_trainer(
        train_x, train_rule_matches_z, mapping_rules_labels_t, config: Dict
) -> NoDenoisingTrainer:
    """Train a logistic regression model; with SimpleDsModelTrainer."""

    params = config.get("hyp_params")
    model = LogisticRegressionModel(params.get('num_features'), mapping_rules_labels_t.shape[1])

    custom_model_config = MajorityConfig(
        model=model, optimizer_=AdamW(model.parameters(), lr=params.get('learning_rate')),
        epochs=params.get('num_epochs'),
        filter_non_labelled=config.get("hyp_params").get("filter_empty_labels")
    )

    trainer = NoDenoisingTrainer(
        model,
        model_input_x=train_x,
        mapping_rules_labels_t=mapping_rules_labels_t,
        rule_matches_z=train_rule_matches_z,
        trainer_config=custom_model_config
    )

    return trainer
