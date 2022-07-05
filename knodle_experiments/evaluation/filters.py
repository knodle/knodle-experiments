import pandas as pd


def filter_test(df: pd.DataFrame, dataset: str = "imdb", allow_majority: bool = True) -> pd.DataFrame:

    if allow_majority:
        allowed_result_types = ["test", "test_majority"]
    else:
        allowed_result_types = ["test"]

    df = df[df["dataset"] == dataset]
    df = df[df["result_type"].isin(allowed_result_types)]

    return df