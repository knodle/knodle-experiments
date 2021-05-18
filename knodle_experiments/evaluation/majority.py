from typing import Dict

import os
import json

import numpy as np
from sklearn.metrics import classification_report

from knodle.transformation.majority import probabilies_to_majority_vote, z_t_matrices_to_majority_vote_probs


def majority_sklearn_report(
        rule_matches_z: np.array, mapping_rules_labels_t: np.array, labels_y: np.array
) -> Dict:
    rule_counts_probs = z_t_matrices_to_majority_vote_probs(rule_matches_z, mapping_rules_labels_t)

    kwargs = {"choose_random_label": True}
    majority_y = np.apply_along_axis(probabilies_to_majority_vote, axis=1, arr=rule_counts_probs, **kwargs)

    sklearn_report = classification_report(labels_y, majority_y, output_dict=True)

    return sklearn_report


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
