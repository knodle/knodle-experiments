import os

from minio_connector.connector import MinioConnector


def download_data_folder(source_path: str, target_path: str):
    c = MinioConnector()
    c.download_dir(source_path, target_path)


def download_unprocessed_imdb_data(target_path):
    download_data_folder("datasets/imdb", target_path)


def download_imdb_data(target_path):
    download_data_folder("datasets/imdb/processed", target_path)


def download_spouse_data(target_path: str):
    download_data_folder("datasets/spouse/processed", target_path)


def download_tacred_data(target_path):
    download_data_folder("datasets/conll", target_path)


def download_spam_data(target_path):
    download_data_folder("datasets/spam/processed", target_path)


def download_dataset(dataset: str, data_dir: str):
    known_dataset = ["imdb", "spam", "tacred", "spouse"]
    if dataset not in known_dataset:
        raise ValueError(f"We only know those datasets: {known_dataset}")

    os.makedirs(data_dir, exist_ok=True)
    if dataset == "imdb":
        download_imdb_data(data_dir)
    elif dataset == "tacred":
        download_tacred_data(data_dir)
    elif dataset == "spam":
        download_spam_data(data_dir)
    elif dataset == "spouse":
        download_spouse_data(data_dir)
