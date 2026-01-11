def calculate_data_quality(profile):
    """
    Returns:
        score (0–100)
        level (text)
        messages (list of warnings)
    """

    score = 100
    messages = []

    total_rows = profile["rows"]
    total_cols = profile["columns"]
    duplicates = profile["duplicates"]
    missing = profile["missing"].sum()

    # -------------------------
    # Missing values penalty
    # -------------------------
    if missing > 0:
        missing_ratio = missing / (total_rows * total_cols)
        penalty = int(missing_ratio * 100)
        score -= min(penalty, 30)
        messages.append(f"Dataset contains {missing} missing values.")

    # -------------------------
    # Duplicate rows penalty
    # -------------------------
    if duplicates > 0:
        dup_ratio = duplicates / total_rows
        penalty = int(dup_ratio * 100)
        score -= min(penalty, 20)
        messages.append(f"{duplicates} duplicate rows detected.")

    # -------------------------
    # Low feature count penalty
    # -------------------------
    if len(profile["numeric_cols"]) < 2:
        score -= 15
        messages.append("Very few numeric features available for modeling.")

    # -------------------------
    # Clamp score
    # -------------------------
    score = max(0, min(100, score))

    # -------------------------
    # Quality label
    # -------------------------
    if score >= 85:
        level = "🟢 Excellent Dataset"
    elif score >= 65:
        level = "🟡 Good Dataset"
    elif score >= 40:
        level = "🟠 Fair Dataset"
    else:
        level = "🔴 Poor Dataset"

    return score, level, messages
