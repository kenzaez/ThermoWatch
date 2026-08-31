# data_source.py
# bloc 1 de l architecture : lit les donnee depuis le csv
# le reste de l app (model_service, anomaly_logic, main) ne sait pas
# QUE ca vient d un csv, il appelle juste ces fonctions
# si un jour on a acces a une base de donnee en direct, ce fichier
# sera le seul a changer, le reste de l app ne bouge pas

import random

import numpy as np
import pandas as pd

CSV_PATH = "../data/processed/all_data_with_all_anomalies.csv"

# variable globale qui garde le csv charge en memoire
# charge une seule fois au demarrage de l api, pas a chaque requete
# (sinon relire un gros csv a chaque appel serait tres lent)
_all_data = None


def load_data():
    # charge le csv en memoire, appele une seule fois au demarrage
    global _all_data
    print("chargement du csv en memoire")
    _all_data = pd.read_csv(CSV_PATH, parse_dates=["Time"])
    _all_data = _all_data.sort_values(["device_id", "Time"]).reset_index(drop=True)

    # --- FILTRE ANTI-BRUIT (fenetre de 3 points) ---
    # recalcule lstm_anomaly : il faut que le flag soit a 1 sur 3 ligne
    # consecutive pour le meme device pour compter comme vraie anomalie
    # (evite de flag une seule minute isolee, du bruit statistique
    # attendu du seuil percentile 90, cf eval_lstm.py chapitre "streak")
    # NOTE : ce filtre s applique seulement aux donnee historique
    # (get_history, get_alerts) - pas a la demo temps reel qui calcule
    # sa propre anomalie via model_service/anomaly_logic
    _all_data["lstm_anomaly"] = (
        _all_data.groupby("device_id")["lstm_anomaly"]
        .rolling(window=3)
        .sum()
        .reset_index(0, drop=True)
        .ge(3)
        .astype(int)
    )
    # -----------------------------------------------

    print(f"csv charge : {len(_all_data)} ligne")


def get_all_devices():
    # renvoi la liste des device avec leur type et sensor group
    # une ligne par device (pas toute l historique)
    devices = (
        _all_data[["device_id", "type", "sensor_group"]]
        .drop_duplicates(subset="device_id")
        .to_dict(orient="records")
    )
    return devices


def get_latest(device_id):
    # renvoi la derniere ligne connue pour un device
    g = _all_data[_all_data["device_id"] == device_id]
    if len(g) == 0:
        return None
    return g.iloc[-1].to_dict()


def get_history(device_id, start=None, end=None, limit=1000):
    # renvoi l historique d un device, avec filtre optionnel de date
    # limit pour eviter de renvoyer des dizaine de milliers de ligne
    # d un coup si personne precise de periode
    g = _all_data[_all_data["device_id"] == device_id]

    if start is not None:
        g = g[g["Time"] >= start]
    if end is not None:
        g = g[g["Time"] <= end]

    g = g.tail(limit).replace({np.nan: None})
    return g.to_dict(orient="records")


def get_random_window(device_id, window_size=61):
    # tire une fenetre de window_size ligne CONSECUTIVE au hasard dans
    # l historique d un device (pas forcement les dernieres ligne)
    # sert pour la demo "temps reel" : sans ca, un meme device donnerait
    # toujours exactement le meme resultat a chaque clic, ce qui n a pas
    # l air "en direct"
    g = _all_data[_all_data["device_id"] == device_id].reset_index(drop=True)

    if len(g) < window_size:
        return []

    max_start = len(g) - window_size
    start_idx = random.randint(0, max_start)

    window = g.iloc[start_idx : start_idx + window_size].replace({np.nan: None})
    return window.to_dict(orient="records")


def get_alerts(method="lstm_anomaly", start=None, end=None):
    # renvoi toute les ligne flag anomaly, tt device confondu
    # method = quelle colonne regarder : lstm_anomaly, baseline_anomaly, zscore_anomaly
    g = _all_data[_all_data[method] == 1]

    if start is not None:
        g = g[g["Time"] >= start]
    if end is not None:
        g = g[g["Time"] <= end]

    # remplace NaN par None, sinon fastapi n arrive pas a serialiser en json
    g = g.replace({np.nan: None})
    return g.to_dict(orient="records")


def get_anomaly_episodes(method="lstm_anomaly", merge_gap_minutes=10, min_duration_minutes=10):
    # regroupe les ligne flag CONSECUTIVE en "episode", PUIS fusionne les
    # episode proches entre eux, PUIS retire les tout petit episode
    #
    # pourquoi la fusion est necessaire : le filtre anti-bruit (rolling
    # window 3 dans load_data) ne fait que ROGNER les bord d un episode,
    # pas le supprimer - un burst de exactement 3 minute d affile devient
    # 1 seule minute flag apres filtre, ce qui compte quand meme comme
    # 1 episode a part entiere. Avec un seuil percentile 90 applique en
    # continu sur des centaine de milliers de ligne, l erreur du LSTM
    # traverse le seuil des milliers de fois naturellement (comportement
    # attendu du percentile, pas un bug) - d ou le besoin de fusionner
    # les episode rapproches en UN SEUL incident, et de filtrer les
    # micro-episode qui restent trop court pour etre significatif
    episodes = []

    for device_id, group in _all_data.groupby("device_id"):
        g = group.sort_values("Time").reset_index(drop=True)
        is_flag = g[method] == 1

        # etape 1 : detecte les bloc de minute flag consecutive
        run_id = (is_flag != is_flag.shift()).cumsum()
        g["_run_id"] = run_id

        raw_episodes = []
        for _, sub in g[is_flag].groupby("_run_id"):
            raw_episodes.append({
                "device_id": device_id,
                "debut": sub["Time"].iloc[0],
                "fin": sub["Time"].iloc[-1],
                "temperature_max": float(sub["Temperature"].max()),
                "temperature_min": float(sub["Temperature"].min()),
            })

        if len(raw_episodes) == 0:
            continue

        # etape 2 : fusionne les episode separe de moins de
        # merge_gap_minutes - probablement le meme incident qui vacille
        # autour du seuil, pas 2 evenement different
        raw_episodes.sort(key=lambda e: e["debut"])
        merged = [raw_episodes[0]]

        for ep in raw_episodes[1:]:
            last = merged[-1]
            gap = (ep["debut"] - last["fin"]).total_seconds() / 60

            if gap <= merge_gap_minutes:
                # fusionne avec le precedent
                last["fin"] = max(last["fin"], ep["fin"])
                last["temperature_max"] = max(last["temperature_max"], ep["temperature_max"])
                last["temperature_min"] = min(last["temperature_min"], ep["temperature_min"])
            else:
                merged.append(ep)

        # etape 3 : calcule la duree finale et retire les episode trop court
        for ep in merged:
            duree = int((ep["fin"] - ep["debut"]).total_seconds() / 60) + 1
            if duree >= min_duration_minutes:
                ep["duree_minutes"] = duree
                episodes.append(ep)

    if len(episodes) == 0:
        return []

    episodes_df = pd.DataFrame(episodes).sort_values("debut", ascending=False)
    episodes_df = episodes_df.replace({np.nan: None})
    return episodes_df.to_dict(orient="records")