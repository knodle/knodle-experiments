import pandas as pd
import numpy as np

from knodle.trainer.utils.denoise import get_majority_vote_probs


def get_majority_stats(
        rule_matches_z: np.array, mapping_rules_labels_t: np.array, labels_y: np.toarray
    ) -> [float, pd.Series, pd.Series]:


    labels_maj = get_majority_vote_probs(
        rule_matches_z, mapping_rules_labels_t
    )
    labels_maj = pd.Series(labels_maj.argmax(axis=1))

    majority_accuracy = (labels_maj == labels_y).mean()
    z_distribution = rule_matches_z.sum(axis=1)
    hit_distribution = labels_maj.value_counts()
    label_distribution = labels_y.value_counts()

    return majority_accuracy, hit_distribution, label_distribution
