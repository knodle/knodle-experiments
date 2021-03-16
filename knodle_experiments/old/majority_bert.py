from typing import Dict

from transformers import AutoModelForSequenceClassification, AdamW

from knodle.trainer.baseline.majority_config import MajorityConfig
from knodle.trainer.baseline.bert import MajorityBertTrainer


def get_majority_bert_trainer(
        train_x, train_rule_matches_z, mapping_rules_labels_t, config: Dict
) -> MajorityBertTrainer:
    """Train a logistic regression model; with SimpleDsModelTrainer."""

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.train()

    custom_model_config = MajorityConfig(
        model=model,
        optimizer_=AdamW(model.parameters(), lr=config.get('hyp_params').get('learning_rate')),
        batch_size=config.get('hyp_params').get('batch_size'),
        epochs=config.get('hyp_params').get('num_epochs'),
        filter_non_labelled=config.get("hyp_params").get("filter_empty_labels")
    )

    trainer = MajorityBertTrainer(
        model,
        mapping_rules_labels_t=mapping_rules_labels_t,
        model_input_x=train_x,
        rule_matches_z=train_rule_matches_z,
        trainer_config=custom_model_config,
    )
    return trainer
