import os
from typing import Dict, List

import json
import pandas as pd

from minio_connector.connector import MinioConnector


def download(source_dir: str, local_dir: str):
    m = MinioConnector()
    m.download_dir(source_dir, target_folder=local_dir)


def load_config(folder: str) -> Dict:
    with open(os.path.join(folder, "config.json"), "r") as f:
        return json.load(f)


def load_json_results(local_dir: str, result_type: str, label_class=None) -> pd.DataFrame:
    supported_types = ["test_majority", "train", "test"]
    if result_type not in supported_types:
        raise ValueError(f"result_type {result_type} must be one of {supported_types}.")

    if label_class is None:
        label_class = "macro avg"
    subfolders = os.listdir(local_dir)

    df = []

    for run_id in subfolders:
        result_file = os.path.join(local_dir, run_id, f"{result_type}_result_dict.json")
        if not os.path.isfile(result_file):
            continue
        config = load_config(os.path.join(local_dir, run_id))
        with open(result_file, "r") as f:
            result = json.load(f)
            if isinstance(result, List):
                result = result[0]
            if "macro_f1" in result:
                continue

            accuracy = round(result.get("accuracy"), 3)
            macro_f1 = round(result.get(label_class).get("f1-score"), 3)
            precision = round(result.get(label_class).get("precision"), 3)
            recall = round(result.get(label_class).get("recall"), 3)
            df.append([
                config.get("dataset"),
                result_type,
                config.get("method"),
                accuracy,
                macro_f1,
                precision,
                recall,
                config.get("hyp_params"),
                run_id
            ])
    df = pd.DataFrame(df, columns=[
        "dataset", "result_type", "method", "accuracy", "f1", "precision", "recall", "hyp_params", "run_id"
    ])

    if result_type == "test_majority":
        df["method"] = "majority_vote"
    return df


def load_test_df(local_dir: str, label_class=None) -> pd.DataFrame:
    return load_json_results(local_dir, "test", label_class=label_class)


def load_train_df(local_dir: str, label_class=None) -> pd.DataFrame:
    return load_json_results(local_dir, "train", label_class=label_class)


def load_majority_test_df(local_dir: str, label_class=None) -> pd.DataFrame:
    return load_json_results(local_dir, "test_majority", label_class=label_class)


def load_full_df(local_dir: str, label_class=None) -> pd.DataFrame:
    dfs = []

    for result_type in ["test_majority", "test"]:
        dfs.append(load_json_results(local_dir, result_type, label_class=label_class))

    df = pd.concat(dfs)
    return df
