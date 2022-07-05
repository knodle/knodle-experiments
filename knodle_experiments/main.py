import os
import logging
from typing import Dict, List
import json
from uuid import uuid4

import torch

from knodle_experiments.data.loaders import load_knodle_format_data
from knodle_experiments.data.download import download_dataset
from knodle_experiments.data.preprocess import preprocess_data

from knodle_experiments.experiments import get_trainer

from knodle_experiments.evaluation.majority import majority_evaluation
from knodle_experiments.evaluation.trainer import trainer_evaluation

logger = logging.getLogger(__name__)


def run_experiment(
        config: Dict, download_data: bool = True, processed_dict: Dict = None, disable_majority_vote: bool = False,
        device=None, result_path: str = None
):
    # Download
    if download_data:
        download_dataset(config.get("dataset"), config.get("data_dir"))

    # Load
    data_dict = load_knodle_format_data(config.get("data_dir"))

    # Preprocess
    if processed_dict is None:
        processed_dict = preprocess_data(config=config, data_dict=data_dict)

    # Initialize
    trainer = get_trainer(config=config, processed_dict=processed_dict, data_dict=data_dict)

    if device is not None:
        trainer.trainer_config.device = device
    # run training
    print(trainer.trainer_config.optimizer)
    trainer.train()

    # perform evaluation
    run_id = str(uuid4())
    if result_path is None:
        result_path = os.path.join(config.get("data_dir"), "results", run_id)
    else:
        result_path = os.path.join(result_path, run_id)
    config["result_path"] = result_path
    config["data_keys"] = list(data_dict.keys())

    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, "config.json"), "w") as f:
        json.dump(config, f)

    # majority_vote
    if disable_majority_vote:
        majority_dicts = {}
    else:
        majority_dicts = majority_evaluation(result_path=result_path, preprocessed_dict=processed_dict)

    # Test trainer
    trainer_eval_dicts = trainer_evaluation(trainer, preprocessed_dict=processed_dict, result_path=result_path)

    return run_id, trainer, majority_dicts, trainer_eval_dicts


def run_experiments(configs: List[Dict], **kwargs):
    runs = {}
    for i, config in enumerate(configs):
        torch.cuda.empty_cache()
        logger.info(f"Experiment number {i}")
        logger.info(f"Configuration: {config}")
        run_id, trainer, majority_dicts, trainer_eval_dicts = run_experiment(config, **kwargs)

        runs[run_id] = config

    return runs
