import pandas as pd

def load_csv(file):
    # Try different encodings to handle all CSV types
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            try:
                file.seek(0)  # reset file pointer
            except:
                pass
            continue

    if df is None:
        raise ValueError("Could not read CSV file. Please check the file format.")

    # Fix numeric-looking strings
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
