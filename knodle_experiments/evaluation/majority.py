from typing import Dict
import os
import json

from knodle.evaluation.majority import majority_sklearn_report


def majority_evaluation(result_path: str, preprocessed_dict: Dict):
    splits = ["train", "test"]

    result_dicts = {}
    for split in splits:
        if preprocessed_dict.get(f"{split}_y", None) is not None:

            majority_result_dict = majority_sklearn_report(
                preprocessed_dict[f"{split}_rule_matches_z"],
                preprocessed_dict["mapping_rules_labels_t"],
                preprocessed_dict[f"{split}_y"].tensors[0].cpu().numpy()
            )
            result_dicts[split] = majority_result_dict
            with open(os.path.join(result_path, f"{split}_majority_result_dict.json"), "w") as f:
                json.dump(majority_result_dict, f)

    return result_dicts