# main.py
# bloc 4 de l architecture : les routes de l api (FastAPI)
# ce fichier ne calcule rien lui meme, il appelle juste data_source,
# model_service et anomaly_logic et renvoi le resultat en json
#
# pour lancer l api :
#   uvicorn main:app --reload
# puis ouvrir http://localhost:8000/docs pour voir/tester tt les endpoint

from typing import Optional
from fastapi import FastAPI, HTTPException

import data_source
import model_service
import anomaly_logic

app = FastAPI(title="anomaly detection api", description="api de detection d anomalie pour capteur IoT (REF/CONG/ETUVE)")


@app.on_event("startup")
def startup():
    # charge tout une seule fois au demarrage de l api, pas a chaque requete
    print("demarrage de l api")
    data_source.load_data()
    model_service.load_models_and_thresholds()
    print("api prete")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "anomaly detection api, voir /docs pour la liste des endpoint",
    }


@app.get("/devices")
def list_devices():
    # renvoi la liste des 9 device avec leur type
    return data_source.get_all_devices()


@app.get("/devices/{device_id}/latest")
def latest_reading(device_id: str):
    # renvoi la derniere lecture connue d un device (temperature + les 3 flag)
    result = data_source.get_latest(device_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"device {device_id} non trouve")
    return result


@app.get("/devices/{device_id}/history")
def history(device_id: str, start: Optional[str] = None, end: Optional[str] = None, limit: int = 1000):
    # renvoi l historique d un device, filtre optionnel de date
    # limit pour eviter de renvoyer des dizaine de milliers de ligne d un coup
    result = data_source.get_history(device_id, start, end, limit)
    if len(result) == 0:
        raise HTTPException(status_code=404, detail=f"device {device_id} non trouve ou pas de donnee sur cette periode")
    return result


@app.get("/alerts")
def alerts(method: str = "lstm_anomaly", start: Optional[str] = None, end: Optional[str] = None):
    # renvoi toute les ligne flag anomalie, tt device confondu
    # method = lstm_anomaly (par defaut), baseline_anomaly, ou zscore_anomaly
    methodes_valide = ["lstm_anomaly", "baseline_anomaly", "zscore_anomaly"]
    if method not in methodes_valide:
        raise HTTPException(status_code=400, detail=f"method doit etre un de : {methodes_valide}")
    return data_source.get_alerts(method, start, end)


@app.get("/alerts/episodes")
def alerts_episodes(
    method: str = "lstm_anomaly",
    merge_gap_minutes: int = 10,
    min_duration_minutes: int = 10,
):
    # renvoi les anomalie regroupee en EPISODE (bloc de minute consecutive,
    # fusionne si proche, filtre si trop court) plutot qu une ligne par
    # minute flag - beaucoup plus lisible pour savoir combien de vrai
    # INCIDENT ont eu lieu, pas combien de minute individuelle
    methodes_valide = ["lstm_anomaly", "baseline_anomaly", "zscore_anomaly"]
    if method not in methodes_valide:
        raise HTTPException(status_code=400, detail=f"method doit etre un de : {methodes_valide}")
    return data_source.get_anomaly_episodes(method, merge_gap_minutes, min_duration_minutes)


@app.post("/devices/{device_id}/predict_demo")
def predict_demo(device_id: str):
    # endpoint BONUS : simule une prediction "temps reel"
    # tire une fenetre de 61 ligne AU HASARD dans l historique du device
    # (60 pour la fenetre, la derniere comme si elle venait d arriver a
    # l instant) plutot que toujours les 61 dernieres - sinon un meme
    # device donnerait toujours exactement le meme resultat a chaque
    # clic, ce qui n a pas l air "en direct" pour la demo
    # (pas de vrai flux live pour l instant, pas d acces a la base de
    # donnee derriere grafana, cf notes projet)
    history_rows = data_source.get_random_window(device_id, window_size=61)
    if len(history_rows) < 61:
        raise HTTPException(status_code=400, detail="pas assez d historique pour ce device")

    window = [row["Temperature"] for row in history_rows[:-1]]
    actual_next = history_rows[-1]["Temperature"]

    result = anomaly_logic.check_anomaly(device_id, window, actual_next)
    if result is None:
        raise HTTPException(status_code=404, detail=f"device {device_id} non trouve ou pas de model pour son type")

    # on rajoute le timestamp tire au hasard dans la reponse, utile
    # pour verifier/afficher a quel moment correspond cette demo
    result["timestamp_simule"] = history_rows[-1]["Time"]

    return result