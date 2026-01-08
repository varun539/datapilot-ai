import pandas as pd


def basic_profile(df):
    profile = {}

    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    # Data types
    profile["dtypes"] = df.dtypes.astype(str)

    # Missing values
    profile["missing"] = df.isnull().sum()

    # Duplicate rows
    profile["duplicates"] = df.duplicated().sum()

    # Numeric & categorical columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    profile["numeric_cols"] = numeric_cols
    profile["categorical_cols"] = categorical_cols

    # Basic statistics
    profile["describe"] = df.describe()

    return profile
