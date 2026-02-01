import json
import datetime

EXPERIMENT_LOG = "experiments.json"

def log_experiment(record):
    record["timestamp"] = datetime.datetime.now().isoformat()

    try:
        with open(EXPERIMENT_LOG, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(record)

    with open(EXPERIMENT_LOG, "w") as f:
        json.dump(data, f, indent=2)

def load_experiments():
    try:
        with open(EXPERIMENT_LOG, "r") as f:
            return json.load(f)
    except:
        return []
