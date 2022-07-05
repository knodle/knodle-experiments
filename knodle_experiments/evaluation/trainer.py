from typing import Dict
import os
import json

from torch.utils.data import TensorDataset

from knodle.trainer.trainer import Trainer


def trainer_evaluation(trainer: Trainer, preprocessed_dict: Dict, result_path: str) -> Dict:
    splits = ["train", "dev", "test"]

    results_dicts = {}
    for split in splits:
        if preprocessed_dict.get(f"{split}_y", None) is not None:
            args = [
                preprocessed_dict.get(f"{split}_x"),
                preprocessed_dict.get(f"{split}_y")
            ]
            results_dict = trainer.test(*args)
            print(results_dict)
            results_dicts[split] = results_dict
            with open(os.path.join(result_path, f"{split}_result_dict.json"), "w") as f:
                json.dump(results_dict, f)

    return results_dicts
