import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def transform_income_ds(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # handle na (this dataset has null categorical only)
    df = df.fillna("Others")

    # target label
    df["income"] = np.where(df["income"] == "<=50K", 0, 1)

    # binary class
    df["male"] = np.where(df["sex"] == "Male", 1, 0)

    # drop bad class
    df = df.drop(["workclass", "native-country"], axis=1)

    # one hot encode categorical
    df = pd.get_dummies(df, dtype=np.float64)

    df = df.rename(columns={"income": "y"})
    return df


def transform_bank_ds(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df = df.rename(columns={"Exited": "y"})
    return df


def get_train_test_ds(ds: str, file_path: str, random_state: int):
    df = None

    if ds == "income":
        df = transform_income_ds(file_path)
    elif ds == "bank":
        df = transform_bank_ds(file_path)
    else:
        raise Exception(f"{ds} unknown")

    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    return X, X_train, X_test, y, y_train, y_test
