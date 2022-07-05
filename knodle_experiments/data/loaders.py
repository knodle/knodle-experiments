import os
from typing import List

from joblib import load
import pandas as pd
import numpy as np


def load_unprocessed_imdb_data(data_dir: str):
    # load into memory
    imdb_dataset = pd.read_csv(os.path.join(data_dir, "imdb_data_preprocessed.csv"))
    rule_matches_z = load(os.path.join(data_dir, "rule_matches.lib"))
    mapping_rules_labels_t = load(os.path.join(data_dir, "mapping_rules_labels.lib"))

    return imdb_dataset, rule_matches_z, mapping_rules_labels_t


def load_plain_conll_data():
    # data location
    data_dir = os.getenv("data_dir", False)
    if not data_dir:
        raise ValueError("Provide a data directory.")

    # load into memory
    conll_dataset = pd.read_csv(os.path.join(data_dir, "imdb_data", "train_samples.csv"))
    rule_matches_z = load(os.path.join(data_dir, "imdb_data", "z_matrix.lib"))
    mapping_rules_labels_t = load(os.path.join(data_dir, "imdb_data", "t_matrix.lib"))

    return conll_dataset, rule_matches_z, mapping_rules_labels_t


def load_imdb_data(data_dir: str):
    # load into memory

    splits = ["train", "dev", "test"]
    data_dict = dict()

    for split in splits:
        df = load(os.path.join(data_dir, f"{split}_df.lib"))
        data_dict[f"{split}_x"] = df['reviews_preprocessed'].tolist().copy()
        data_dict[f"{split}_y"] = df["label_id"].values.copy()

        data_dict[f"{split}_rule_matches_z"] = load(os.path.join(data_dir, f"{split}_rule_matches_z.lib"))

    data_dict["mapping_rules_labels_t"] = load(os.path.join(data_dir, "mapping_rules_labels_t.lib"))

    return data_dict


def load_tacred_data(data_dir: str):
    splits = ["train", "dev", "test"]
    data_dict = dict()

    for split in splits:
        df = load(os.path.join(data_dir, f"df_{split}.lib"))
        data_dict[f"{split}_x"] = df["samples"].tolist().copy()
        if split != 'train':
            data_dict[f"{split}_y"] = df["labels"].values

        data_dict[f"{split}_rule_matches_z"] = load(os.path.join(data_dir, f"{split}_rule_matches_z.lib"))

    data_dict["mapping_rules_labels_t"] = load(os.path.join(data_dir, "mapping_rules_labels.lib"))

    return data_dict


def load_spouse_data(data_dir: str) -> [List[str], np.array, np.array, np.array, List[str], np.array]:
    splits = ["train", "dev", "test"]
    data_dict = dict()

    for split in splits:
        df = load(os.path.join(data_dir, f"df_{split}.lib"))
        data_dict[f"{split}_x"] = df["sentence"].tolist().copy()

        file_name_y = os.path.join(data_dir, f"Y_{split}.lib")
        if os.path.isfile(file_name_y):
            data_dict[f"{split}_y"] = load(file_name_y)

        data_dict[f"{split}_rule_matches_z"] = load(os.path.join(data_dir, f"{split}_rule_matches_z.lib"))

    data_dict["mapping_rules_labels_t"] = load(os.path.join(data_dir, "mapping_rules_labels_t.lib"))

    return data_dict


def load_spam_data(data_dir: str) -> [List[str], np.array, np.array, np.array, List[str], np.array]:
    splits = ["train", "test"]
    data_dict = dict()

    for split in splits:
        df = load(os.path.join(data_dir, f"df_{split}.lib"))
        data_dict[f"{split}_x"] = df["text"].tolist().copy()

        file_name_y = os.path.join(data_dir, f"Y_{split}.lib")
        if os.path.isfile(file_name_y):
            data_dict[f"{split}_y"] = load(file_name_y)

        data_dict[f"{split}_rule_matches_z"] = load(os.path.join(data_dir, f"{split}_rule_matches_z.lib"))

    data_dict["mapping_rules_labels_t"] = load(os.path.join(data_dir, "mapping_rules_labels.lib"))

    return data_dict


def load_knodle_format_data(data_dir: str, splits=["train", "dev", "test"], text_col: str = "sample",
                            label_col: str = "label_id"):
    data_dict = dict()

    for split in splits:
        df_path = os.path.join(data_dir, f"df_{split}.csv")
        if not os.path.isfile(df_path):
            continue
        df = pd.read_csv(df_path, sep=";")
        data_dict[f"{split}_x"] = df[text_col].tolist().copy()

        if label_col in df.columns:
            data_dict[f"{split}_y"] = df[label_col].values

        data_dict[f"{split}_rule_matches_z"] = load(os.path.join(data_dir, f"{split}_rule_matches_z.lib"))

    data_dict["mapping_rules_labels_t"] = load(os.path.join(data_dir, "mapping_rules_labels_t.lib"))

    return data_dict


def load_dataset(dataset: str, data_dir: str):
    known_dataset = ["imdb", "tacred", "spouse", "spam"]
    if dataset not in known_dataset:
        raise ValueError(f"We only know those datasets: {known_dataset}")

    if dataset == "imdb":
        return load_imdb_data(data_dir)
    elif dataset == "tacred":
        return load_tacred_data(data_dir)
    elif dataset == "spouse":
        return load_spouse_data(data_dir)
    elif dataset == "spam":
        return load_spam_data(data_dir)
    elif dataset == "sms":
        return load_knodle_format_data(data_dir, text_col="sample", label_col="label")
    elif dataset == "trec":
        return load_knodle_format_data(data_dir, text_col="sample", label_col="label")
