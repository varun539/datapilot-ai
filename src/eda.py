import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_profile(df):
    profile = {}

    df_copy = df.copy()
    datetime_cols = []

    # -----------------------------
    # Robust datetime detection
    # -----------------------------
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            parsed = pd.to_datetime(
                df_copy[col],
                errors="coerce",
                dayfirst=True,
                infer_datetime_format=True
            )

            if parsed.notna().mean() > 0.7:
                df_copy[col] = parsed
                datetime_cols.append(col)

    numeric_cols = df_copy.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=["object"]).columns.tolist()

    # -----------------------------
    # 🚮 REMOVE ID / USELESS COLUMNS FROM PROFILE
    # -----------------------------
    id_keywords = ["id", "uuid", "index", "code", "number"]

    def is_id_column(col):
        name = col.lower()
        if any(k in name for k in id_keywords):
            return True
        if df_copy[col].nunique() / len(df_copy) > 0.98:
            return True
        return False

    numeric_cols = [c for c in numeric_cols if not is_id_column(c)]
    categorical_cols = [c for c in categorical_cols if not is_id_column(c)]

    # -----------------------------
    # Build profile
    # -----------------------------
    profile["rows"] = df_copy.shape[0]
    profile["columns"] = df_copy.shape[1]
    profile["duplicates"] = df_copy.duplicated().sum()
    profile["dtypes"] = df_copy.dtypes
    profile["missing"] = df_copy.isnull().sum()

    profile["numeric_cols"] = numeric_cols
    profile["categorical_cols"] = categorical_cols
    profile["datetime_cols"] = datetime_cols
    profile["describe"] = df_copy[numeric_cols].describe()

    return profile




def plot_numeric_distributions(df, numeric_cols):
    figures = []

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))   # 👈 smaller
        df[col].hist(bins=20, ax=ax)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        figures.append(fig)

    return figures


def plot_correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(4.5, 3))   # 👈 smaller
    sns.heatmap(corr, cmap="coolwarm", ax=ax, cbar=False)
    ax.set_title("Correlation", fontsize=10)
    ax.tick_params(labelsize=7)
    plt.tight_layout()

    return fig



def plot_categorical_counts(df, categorical_cols, top_n=8):
    figures = []

    for col in categorical_cols:
        vc = df[col].value_counts().head(top_n)
        labels = vc.index.astype(str).str.slice(0, 25)

        fig, ax = plt.subplots(figsize=(4.5, 2.8))   # 👈 compact
        ax.barh(labels, vc.values)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.invert_yaxis()
        plt.tight_layout()

        figures.append(fig)

    return figures



def plot_time_series(df, datetime_cols, numeric_cols):
    if not datetime_cols or not numeric_cols:
        return None, None

    date_col = datetime_cols[0]

    # Choose SALES-like column automatically
    value_col = None
    for col in numeric_cols:
        if "sales" in col.lower() or "revenue" in col.lower():
            value_col = col
            break

    if value_col is None:
        value_col = numeric_cols[0]

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(
        temp_df[date_col], errors="coerce", dayfirst=True
    )
    temp_df = temp_df.dropna(subset=[date_col])

    # Monthly aggregation
    temp_df["year_month"] = temp_df[date_col].dt.to_period("M")
    monthly = temp_df.groupby("year_month")[value_col].mean().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)

    # Yearly aggregation
    temp_df["year"] = temp_df[date_col].dt.year
    yearly = temp_df.groupby("year")[value_col].mean().reset_index()

    # Monthly plot
    fig_month, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(monthly["year_month"], monthly[value_col])
    ax1.set_title(f"{value_col} - Monthly Trend")
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Yearly plot
    fig_year, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(yearly["year"], yearly[value_col], marker="o")
    ax2.set_title(f"{value_col} - Yearly Trend")
    plt.tight_layout()

    return fig_month, fig_year
