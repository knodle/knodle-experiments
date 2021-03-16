from typing import Dict, List
import logging

from transformers import AutoTokenizer

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def create_tfidf_input(
        data_dict: Dict,
        num_features: int = 2000,
) -> Dict:
    """Transform data for trianing purposes.

    :param data_dict: Dictionary holding data
    :param num_features: Number of features
    :return: Data needed for an arbitrary training run based on tf-idf
    """
    if "train_x" not in data_dict:
        raise ValueError("")

    logger.info(f"Starting tfidf preprocessing with {num_features} features.")

    train_x = data_dict.get("train_x")
    vectorizer = CountVectorizer(max_features=num_features)
    vectorizer = vectorizer.fit(train_x)

    splits = ["train", "dev", "test"]
    preprocessed_dict = {
        "mapping_rules_labels_t": data_dict.get("mapping_rules_labels_t")
    }

    for split in splits:
        x_name = f"{split}_x"
        if x_name in data_dict:
            split_x = vectorizer.transform(data_dict.get(x_name))
            split_x = split_x.toarray().astype(np.float)
            split_x = TensorDataset(torch.from_numpy(split_x))
            preprocessed_dict[x_name] = split_x

        rule_name = f"{split}_rule_matches_z"
        if rule_name in data_dict:
            z_matrix = data_dict.get(rule_name)
            if not isinstance(z_matrix, np.ndarray):
                z_matrix = z_matrix
            preprocessed_dict[rule_name] = z_matrix

        y_name = f"{split}_y"
        if y_name in data_dict:
            split_y = data_dict.get(y_name)
            preprocessed_dict[y_name] = TensorDataset(torch.from_numpy(split_y))

    return preprocessed_dict


def convert_text_to_transformer_input(tokenizer, texts: List[str]) -> TensorDataset:
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding.get('input_ids')
    attention_mask = encoding.get('attention_mask')

    input_values_x = TensorDataset(input_ids, attention_mask)

    return input_values_x


def create_transformer_input(
        model_name: str, data_dict: Dict
) -> Dict:
    """Note that this can also be used for DistillBert and other versions.

    :param tokenizer: An aribrary tokenizer from the transformers library.
    :return: Data relevant for BERT training.
    """
    preprocessed_data_dict = {
        "mapping_rules_labels_t": data_dict.get("mapping_rules_labels_t")
    }
    splits = ["train", "dev", "test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for split in splits:
        if f"{split}_x" not in data_dict:
            continue
        ids_mask = convert_text_to_transformer_input(tokenizer, data_dict.get(f"{split}_x"))
        preprocessed_data_dict[f"{split}_x"] = ids_mask

        split_rule_matches_z = data_dict.get(f"{split}_rule_matches_z")
        if split_rule_matches_z is not None:
            preprocessed_data_dict[f"{split}_rule_matches_z"] = split_rule_matches_z

        split_y = data_dict.get(f"{split}_y")
        if split_y is not None:
            split_y = TensorDataset(torch.from_numpy(split_y))
            preprocessed_data_dict[f"{split}_y"] = split_y

    return preprocessed_data_dict


def preprocess_data(config: Dict, data_dict: Dict):
    if config.get("model").get("type") == "logistic_regression":
        processed_dict = create_tfidf_input(data_dict, config.get("hyp_params").get("num_features"))
    elif config.get("model").get("type") == "transformer":
        processed_dict = create_transformer_input(
            config.get("model").get("name"), data_dict
        )
    else:
        raise ValueError("The given method is not specified")

    return processed_dict
