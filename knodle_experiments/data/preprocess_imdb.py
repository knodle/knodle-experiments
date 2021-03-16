import os

import joblib
from sklearn.model_selection import train_test_split

from knodle_experiments.data.download import download_unprocessed_imdb_data
from knodle_experiments.data.loaders import load_unprocessed_imdb_data


def imdb_raw_to_split(data_source_dir: str, data_target_dir):
    # Prepare paths
    os.makedirs(data_target_dir, exist_ok=True)

    # load data
    imdb_dataset, rule_matches_z, mapping_rules_labels_t = load_unprocessed_imdb_data(data_source_dir)

    # split and transform
    DEV_SIZE = 0.1
    TEST_SIZE = 0.1
    RANDOM_STATE = 123

    rest_df, dev_df = train_test_split(imdb_dataset, test_size=DEV_SIZE, random_state=RANDOM_STATE)
    train_df, test_df = train_test_split(rest_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_rule_matches_z = rule_matches_z[train_df.index]
    dev_rule_matches_z = rule_matches_z[dev_df.index]
    test_rule_matches_z = rule_matches_z[test_df.index]

    # save data
    joblib.dump(train_df, os.path.join(data_target_dir, "train_df.lib"))
    joblib.dump(dev_df, os.path.join(data_target_dir, "dev_df.lib"))
    joblib.dump(test_df, os.path.join(data_target_dir, "test_df.lib"))

    joblib.dump(train_rule_matches_z, os.path.join(data_target_dir, "train_rule_matches_z.lib"))
    joblib.dump(dev_rule_matches_z, os.path.join(data_target_dir, "dev_rule_matches_z.lib"))
    joblib.dump(test_rule_matches_z, os.path.join(data_target_dir, "test_rule_matches_z.lib"))

    joblib.dump(mapping_rules_labels_t, os.path.join(data_target_dir, "mapping_rules_labels_t.lib"))


def full_imdb_splitting(data_dir: str):
    processed_dir = os.path.join(data_dir, "processed")

    download_unprocessed_imdb_data(data_dir)
    imdb_raw_to_split(data_dir, processed_dir)
