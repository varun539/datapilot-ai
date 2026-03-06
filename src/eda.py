import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np

# ======================================================
# GLOBAL STYLE — Dark Premium Theme
# ======================================================
DARK_BG    = "#0E1117"
CARD_BG    = "#1A1D27"
ACCENT     = "#4F8EF7"
ACCENT2    = "#A259FF"
GREEN      = "#00C49A"
RED        = "#FF4B6E"
TEXT       = "#E0E0E0"
GRID       = "#2A2D3A"

def set_style():
    mpl.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "axes.titlesize":    13,
        "axes.labelsize":    10,
        "axes.grid":         True,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        TEXT,
        "font.family":       "sans-serif",
        "figure.dpi":        130,
    })

# ======================================================
# BASIC PROFILE
# ======================================================
def basic_profile(df):
    profile = {}
    df_copy = df.copy()
    datetime_cols = []

    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            parsed = pd.to_datetime(
                df_copy[col], errors="coerce",
                dayfirst=True, infer_datetime_format=True
            )
            if parsed.notna().mean() > 0.7:
                df_copy[col] = parsed
                datetime_cols.append(col)

    numeric_cols     = df_copy.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_copy.select_dtypes(include=["object"]).columns.tolist()

    id_keywords = ["id", "uuid", "index", "code", "number"]

    def is_id_column(col):
        name = col.lower()
        if any(k in name for k in id_keywords):
            return True
        if len(df_copy) > 0 and df_copy[col].nunique() / len(df_copy) > 0.98:
            return True
        return False

    numeric_cols     = [c for c in numeric_cols     if not is_id_column(c)]
    categorical_cols = [c for c in categorical_cols if not is_id_column(c)]

    profile["rows"]            = df_copy.shape[0]
    profile["columns"]         = df_copy.shape[1]
    profile["duplicates"]      = df_copy.duplicated().sum()
    profile["dtypes"]          = df_copy.dtypes
    profile["missing"]         = df_copy.isnull().sum()
    profile["numeric_cols"]    = numeric_cols
    profile["categorical_cols"]= categorical_cols
    profile["datetime_cols"]   = datetime_cols
    profile["describe"]        = df_copy[numeric_cols].describe() if numeric_cols else pd.DataFrame()

    return profile

# ======================================================
# NUMERIC DISTRIBUTIONS — Histogram + KDE
# ======================================================
def plot_numeric_distributions(df, numeric_cols):
    set_style()
    figures = []

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(5, 3))

        data = df[col].dropna()

        # Histogram
        n, bins, patches = ax.hist(
            data, bins=30,
            color=ACCENT, alpha=0.75, edgecolor=DARK_BG, linewidth=0.4
        )

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_vals = kde(x_range)
            # Scale KDE to histogram height
            scale = n.max() / kde_vals.max()
            ax.plot(x_range, kde_vals * scale, color=ACCENT2, linewidth=2, label="KDE")
        except Exception:
            pass

        # Mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color=GREEN, linewidth=1.5, linestyle="--", label=f"Mean: {mean_val:.1f}")

        ax.set_title(f"Distribution — {col}", fontweight="bold", pad=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, framealpha=0.3)
        fig.patch.set_facecolor(DARK_BG)
        plt.tight_layout()
        figures.append(fig)

    return figures

# ======================================================
# CORRELATION HEATMAP — Premium
# ======================================================
def plot_correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) < 2:
        return None

    set_style()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(max(5, len(numeric_cols) * 0.9), max(4, len(numeric_cols) * 0.7)))

    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "color": TEXT},
        linewidths=0.5,
        linecolor=DARK_BG,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.6}
    )

    ax.set_title("Feature Correlation Matrix", fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.patch.set_facecolor(DARK_BG)
    plt.tight_layout()

    return fig

# ======================================================
# CATEGORICAL BAR CHARTS — Horizontal Premium
# ======================================================
def plot_categorical_counts(df, categorical_cols, top_n=8):
    set_style()
    figures = []

    # Color gradient
    colors = [ACCENT, ACCENT2, GREEN, RED, "#F7B731", "#26de81", "#fd9644", "#a29bfe"]

    for col in categorical_cols:
        vc     = df[col].value_counts().head(top_n)
        labels = vc.index.astype(str).str.slice(0, 30)
        values = vc.values

        fig, ax = plt.subplots(figsize=(6, max(3, len(labels) * 0.45)))

        bars = ax.barh(
            labels, values,
            color=colors[:len(labels)],
            edgecolor=DARK_BG,
            linewidth=0.5,
            height=0.65
        )

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center", ha="left",
                fontsize=8, color=TEXT
            )

        ax.set_title(f"Top Values — {col}", fontweight="bold", pad=10)
        ax.set_xlabel("Count")
        ax.invert_yaxis()
        ax.set_xlim(0, values.max() * 1.15)
        fig.patch.set_facecolor(DARK_BG)
        plt.tight_layout()
        figures.append(fig)

    return figures

# ======================================================
# TIME SERIES — Monthly + Yearly Trends
# ======================================================
def plot_time_series(df, datetime_cols, numeric_cols):
    if not datetime_cols or not numeric_cols:
        return None, None

    set_style()
    date_col = datetime_cols[0]

    # Auto-detect sales/revenue column
    value_col = None
    for col in numeric_cols:
        if any(k in col.lower() for k in ["sales", "revenue", "profit", "amount"]):
            value_col = col
            break
    if value_col is None:
        value_col = numeric_cols[0]

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce", dayfirst=True)
    temp_df = temp_df.dropna(subset=[date_col])

    # Monthly
    temp_df["year_month"] = temp_df[date_col].dt.to_period("M")
    monthly = temp_df.groupby("year_month")[value_col].mean().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)

    # Yearly
    temp_df["year"] = temp_df[date_col].dt.year
    yearly = temp_df.groupby("year")[value_col].mean().reset_index()

    # ── Monthly Plot ──────────────────────────────────
    fig_month, ax1 = plt.subplots(figsize=(8, 3.5))

    ax1.fill_between(
        range(len(monthly)),
        monthly[value_col],
        alpha=0.15, color=ACCENT
    )
    ax1.plot(
        range(len(monthly)),
        monthly[value_col],
        color=ACCENT, linewidth=2, marker="o",
        markersize=3, markerfacecolor=ACCENT2
    )

    # Highlight max month
    max_idx = monthly[value_col].idxmax()
    ax1.scatter(max_idx, monthly[value_col].iloc[max_idx],
                color=GREEN, s=80, zorder=5, label=f"Peak: {monthly[value_col].iloc[max_idx]:.0f}")

    step = max(1, len(monthly) // 12)
    ax1.set_xticks(range(0, len(monthly), step))
    ax1.set_xticklabels(monthly["year_month"].iloc[::step], rotation=45, ha="right", fontsize=7)
    ax1.set_title(f"{value_col} — Monthly Trend", fontweight="bold", pad=10)
    ax1.set_ylabel(f"Avg {value_col}")
    ax1.legend(fontsize=8, framealpha=0.3)
    fig_month.patch.set_facecolor(DARK_BG)
    plt.tight_layout()

    # ── Yearly Plot ───────────────────────────────────
    fig_year, ax2 = plt.subplots(figsize=(6, 3))

    bar_colors = [ACCENT if v < yearly[value_col].max() else GREEN for v in yearly[value_col]]

    bars = ax2.bar(
        yearly["year"].astype(str),
        yearly[value_col],
        color=bar_colors,
        edgecolor=DARK_BG,
        linewidth=0.5,
        width=0.55
    )

    for bar, val in zip(bars, yearly[value_col]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + yearly[value_col].max() * 0.01,
            f"{val:.0f}",
            ha="center", va="bottom",
            fontsize=8, color=TEXT
        )

    ax2.set_title(f"{value_col} — Yearly Average", fontweight="bold", pad=10)
    ax2.set_ylabel(f"Avg {value_col}")
    fig_year.patch.set_facecolor(DARK_BG)
    plt.tight_layout()

    return fig_month, fig_year
