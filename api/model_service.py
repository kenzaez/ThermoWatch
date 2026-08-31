# model_service.py
# bloc 2 de l architecture : charge les model lstm et les seuil par device
# lit un fichier json deja calcule par eval_lstm.py (models/device_thresholds.json)
# au lieu de tout recalculer a chaque demarrage de l api (ca prenait
# plusieurs minute avant, maintenant c est quasi instantane)

import json
import os
from pathlib import Path

import numpy as np
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FOLDER = BASE_DIR / "models"
TYPES = ["ETUVE", "REF", "CONG"]
CACHE_PATH = MODEL_FOLDER / "device_thresholds.json"

# variable globale, rempli une seule fois au demarrage de l api
_models = {}
_device_stats = {}
_device_thresholds = {}
_device_type_map = {}


def load_models_and_thresholds():
    # charge les 3 model, et lit les seuil/stat deja calcule par eval_lstm.py
    global _models, _device_stats, _device_thresholds, _device_type_map

    print("chargement des model lstm")
    for device_type in TYPES:
        _models[device_type] = keras.models.load_model(str(MODEL_FOLDER / f"lstm_{device_type}.keras"))
        print(f"model {device_type} charge")

    if not os.path.exists(CACHE_PATH):
        # pas de cache trouve, il faut d abord lancer eval_lstm.py une fois
        # (il genere ce fichier a la fin)
        raise FileNotFoundError(
            f"{CACHE_PATH} introuvable. "
            "Lance eval_lstm.py une fois pour generer les seuil et stat avant de demarrer l api."
        )

    print(f"chargement du cache : {CACHE_PATH}")
    with open(CACHE_PATH, "r") as f:
        cache = json.load(f)

    _device_thresholds.update(cache["thresholds"])
    _device_type_map.update(cache["device_types"])
    # device_stats sauvegarde comme liste [mean, std], on le remet en tuple
    _device_stats.update({dev: tuple(stat) for dev, stat in cache["device_stats"].items()})

    print(f"seuil et stat charge pour {len(_device_thresholds)} device")


def get_device_type(device_id):
    return _device_type_map.get(device_id)


def get_device_stats(device_id):
    # renvoi (mean, std) utilise pour normaliser ce device
    return _device_stats.get(device_id)


def get_threshold(device_id):
    return _device_thresholds.get(device_id)


def predict_next(device_id, temperature_window):
    # temperature_window = liste de SEQ_LENGTH valeur de temperature BRUTE
    # (pas normalise), dans l ordre chronologique
    # renvoi la prediction du model pour la minute suivante, en temperature brute
    device_type = _device_type_map.get(device_id)
    if device_type is None:
        return None

    mean, std = _device_stats[device_id]
    values_norm = (np.array(temperature_window) - mean) / std

    X = values_norm.reshape((1, len(temperature_window), 1))
    model = _models[device_type]

    prediction_norm = model.predict(X, verbose=0)[0][0]
    prediction_temp = prediction_norm * std + mean

    return {
        "prediction_normalisee": float(prediction_norm),
        "prediction_temperature": float(prediction_temp),
    }