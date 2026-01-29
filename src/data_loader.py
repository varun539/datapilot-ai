import pandas as pd

def load_csv(file):
    df = pd.read_csv(file)

    # 🔧 FIX: convert numeric-looking strings to numbers
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df
