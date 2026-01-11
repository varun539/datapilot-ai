import json
import os
from datetime import datetime

REGISTRY_PATH = "models/registry.json"
VERSIONS_DIR = "models/versions"


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


def register_model(
    model_path,
    model_name,
    cv_score,
    feature_count,
    best_params=None
):
    os.makedirs(VERSIONS_DIR, exist_ok=True)

    registry = load_registry()

    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    record = {
        "version_id": version_id,
        "model_name": model_name,
        "model_path": model_path,
        "cv_score": round(float(cv_score), 4),
        "feature_count": feature_count,
        "best_params": best_params or {},
        "created_at": datetime.now().isoformat()
    }

    registry.append(record)
    save_registry(registry)

    return record


def get_all_models():
    return load_registry()
