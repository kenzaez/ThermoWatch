# anomaly_logic.py
# bloc 3 de l architecture : decide si une lecture est une anomalie
# utilise le bloc 2 (model_service) pour predire, mais ne sait pas
# comment le model marche a l interieur, juste "je donne une fenetre,
# on me redonne une prediction"

try:
    from api import model_service
except ModuleNotFoundError:
    import model_service


def check_anomaly(device_id, temperature_window, actual_next_value):
    # temperature_window = les 60 dernieres minute de temperature (brute)
    # actual_next_value = la vraie valeur suivante, pour comparer a la prediction
    #
    # renvoi un dictionnaire avec la prediction, l erreur, et si c est
    # flag anomalie ou pas (compare au seuil de CE device)
    stats = model_service.get_device_stats(device_id)
    threshold = model_service.get_threshold(device_id)

    if stats is None or threshold is None:
        return None

    mean, std = stats

    prediction = model_service.predict_next(device_id, temperature_window)
    if prediction is None:
        return None

    # erreur calculee en espace normalise, comme dans eval_lstm.py
    # (le seuil a ete calcule sur de l erreur normalisee, donc il faut
    # comparer des erreur normalisee, pas des erreur en degre brute)
    actual_next_norm = (actual_next_value - mean) / std
    error_norm = abs(actual_next_norm - prediction["prediction_normalisee"])

    is_anomaly = error_norm > threshold

    return {
        "device_id": device_id,
        "prediction_temperature": prediction["prediction_temperature"],
        "valeur_reelle": actual_next_value,
        "erreur_normalisee": float(error_norm),
        "seuil_device": float(threshold),
        "anomaly": bool(is_anomaly),
    }
