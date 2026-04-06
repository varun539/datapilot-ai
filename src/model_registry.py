import json
import os

FILE = "models_registry.json"

def register_model(model):
    data = []

    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            data = json.load(f)

    data.append(model)

    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_all_models():
    if not os.path.exists(FILE):
        return []

    with open(FILE, "r") as f:
        return json.load(f)
