import pandas as pd


def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df
