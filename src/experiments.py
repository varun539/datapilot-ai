import json
import os

FILE = "experiments.json"

def save_experiment(exp):
    data = []

    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            data = json.load(f)

    data.append(exp)

    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_experiments():
    if not os.path.exists(FILE):
        return []

    with open(FILE, "r") as f:
        return json.load(f)
