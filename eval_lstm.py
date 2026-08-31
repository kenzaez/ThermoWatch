# eval_lstm.py
# charge les 3 model lstm deja entraine
# reconstruit des sequence sur TOUTE les donnee (excursion incluse cette
# fois, contrairement a lstm_prep.py) pour pouvoir evaluer sur les pic connu
# calcule lstm_anomaly et fusionne avec baseline_anomaly / zscore_anomaly

import json
import numpy as np
import pandas as pd
from tensorflow import keras

from lstm_prep import (
    IN_PATH,
    excursion_windows,
    remove_excursions,
    get_device_norm_stats,
    apply_normalization,
    build_sequences_for_type,
)

DATA_FOLDER = "data/processed"
MODEL_FOLDER = "models"

TYPES = ["ETUVE", "REF", "CONG"]

# percentile utilise pour le seuil d anomalie
# calcule seulement sur l erreur HORS excursion connu (sinon les gros
# pic faussent le percentile vers le haut et le seuil devient trop large)
# percentile 90 utilise pour la RECHERCHE (comparer les 3 methode,
# detecter les 2 pic connu meme discret) - trop permissif pour un
# dashboard de monitoring continu, ou il flag ~10% du temps par
# definition mathematique du percentile
# percentile 99 = seuil plus strict, ~1% du temps considere anormal,
# plus adapte a un usage "alerte" plutot que "recherche"
ERROR_PERCENTILE = 99

# timestamp exact du pic pour chaque device (pas juste la fenetre large)
# sert a verifier si les methode detecte vraiment LE pic, pas juste
# du bruit ailleurs dans la semaine
peak_timestamps = {
    "44B0C6": "2026-06-23 12:45",
    "5F6A6A": "2026-07-16 14:02",
}

# demi largeur de la fenetre autour du pic exact, en minute
PEAK_WINDOW_MINUTES = 30


def load_model(device_type):
    # charge un model lstm deja entraine
    model_path = f"{MODEL_FOLDER}/lstm_{device_type}.keras"
    model = keras.models.load_model(model_path)
    print(f"model {device_type} charge")
    return model


def is_in_excursion(device_id_array, time_array, excursion_windows):
    # renvoi un array bool, True si la ligne est dans une fenetre d excursion connu
    mask = np.zeros(len(device_id_array), dtype=bool)
    times = pd.to_datetime(time_array)

    for dev, (start, end) in excursion_windows.items():
        is_dev = device_id_array == dev
        is_win = (times >= start) & (times <= end)
        mask = mask | (is_dev & is_win)

    return mask


def longest_flag_streak(flags):
    # calcule la plus longue serie de 1 d affile dans un array de 0/1
    # sert a voir si les flag lstm sont groupe (vrai signal) ou eparpille
    # (juste du bruit statistique attendu, genre le percentile 90)
    max_streak = 0
    current_streak = 0

    for val in flags:
        if val == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def print_streak_details(merged, device_id, margin_minutes=10):
    # retrouve exactement ou se trouve la plus longue serie flag
    # et affiche la temperature autour, pour comprendre ce qui se passe
    g = merged[merged["device_id"] == device_id].sort_values("Time").reset_index(drop=True)
    flags = g["lstm_anomaly"].values

    best_start = 0
    best_length = 0
    current_start = None
    current_length = 0

    for i, val in enumerate(flags):
        if val == 1:
            if current_start is None:
                current_start = i
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            current_start = None
            current_length = 0

    best_end = best_start + best_length - 1

    print(f"device {device_id}")
    print(f"plus longue serie : {best_length} minute")
    print(f"debut : {g['Time'].iloc[best_start]}")
    print(f"fin : {g['Time'].iloc[best_end]}")
    print("")

    # affiche un peu de contexte avant/apres pour voir la transition
    window_start = max(0, best_start - margin_minutes)
    window_end = min(len(g) - 1, best_end + margin_minutes)
    g_window = g.iloc[window_start:window_end + 1]

    print(g_window[["Time", "Temperature", "lstm_error", "lstm_anomaly"]].to_string(index=False))
    print("")


def main():
    print("debut evaluation lstm")
    print("")

    print("chargement du csv complet (excursion incluse)")
    all_data = pd.read_csv(IN_PATH, parse_dates=["Time"])

    print("recalcul des stat de normalisation (sans excursion, comme en training)")
    df_no_excursion = remove_excursions(all_data, excursion_windows)
    device_stats = get_device_norm_stats(df_no_excursion)

    print("application de la normalisation sur TOUTE les donnee (excursion incluse)")
    all_data_norm = apply_normalization(all_data, device_stats)
    print("")

    all_lstm_frames = []

    # dictionnaire qui va garder le seuil et le type de chaque device
    # sauvegarde en json a la fin, pour que l api n ait pas besoin de
    # tout recalculer a chaque demarrage
    all_thresholds = {}
    all_device_types = {}

    for device_type in TYPES:
        print(f"--- type {device_type} ---")
        model = load_model(device_type)

        # reconstruit les sequence sur TOUTE les donnee de ce type
        # (contrairement a lstm_prep.py, les excursion sont gardee ici
        # expres, pour pouvoir evaluer le modele dessus)
        X, y, y_time, y_device = build_sequences_for_type(all_data_norm, device_type)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        y_pred = model.predict(X, verbose=0)
        error = np.abs(y.reshape(-1) - y_pred.reshape(-1))

        excursion_mask = is_in_excursion(y_device, y_time, excursion_windows)

        # seuil calcule PAR DEVICE, pas par type
        # sinon un device plus bruyant que les autre (ex BEKO / 44B0C6)
        # se retrouve compare a un seuil commun calibre sur des device
        # plus calme, et ses propres pic se noie dans son bruit normal
        anomaly_flag = np.zeros(len(error), dtype=int)

        for dev in np.unique(y_device):
            is_dev = y_device == dev
            is_dev_normal = is_dev & (~excursion_mask)

            dev_normal_error = error[is_dev_normal]
            dev_threshold = np.percentile(dev_normal_error, ERROR_PERCENTILE)

            anomaly_flag[is_dev] = (error[is_dev] > dev_threshold).astype(int)

            print(f"device {dev} : seuil {dev_threshold:.6f}")

            # garde le seuil et le type de ce device, pour sauvegarde json
            # a la fin (evite de tout recalculer a chaque demarrage de l api)
            all_thresholds[dev] = float(dev_threshold)
            all_device_types[dev] = device_type

        print(f"nb anomalie flag (toute donnee) : {anomaly_flag.sum()} / {len(anomaly_flag)}")
        print(f"nb anomalie flag DANS les fenetre d excursion connu : {anomaly_flag[excursion_mask].sum()} / {excursion_mask.sum()}")

        df_type = pd.DataFrame({
            "device_id": y_device,
            "Time": pd.to_datetime(y_time),
            "lstm_error": error,
            "lstm_anomaly": anomaly_flag,
        })
        all_lstm_frames.append(df_type)
        print("")

    # sauvegarde les seuil, le type de chaque device, et les stat de
    # normalisation dans un json - permet a l api de demarrer sans
    # tout recalculer a chaque fois (calcul deja fait ici, une seule fois)
    cache_path = f"{MODEL_FOLDER}/device_thresholds.json"
    cache = {
        "thresholds": all_thresholds,
        "device_types": all_device_types,
        "device_stats": {dev: list(stat) for dev, stat in device_stats.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"seuil et stat sauvegarde dans : {cache_path}")
    print("")

    lstm_results = pd.concat(all_lstm_frames, ignore_index=True)

    merged = all_data.merge(
        lstm_results[["device_id", "Time", "lstm_error", "lstm_anomaly"]],
        on=["device_id", "Time"],
        how="left",
    )
    merged["lstm_anomaly"] = merged["lstm_anomaly"].fillna(0).astype(int)

    out_path = f"{DATA_FOLDER}/all_data_with_all_anomalies.csv"
    merged.to_csv(out_path, index=False)
    print(f"csv final sauvegarde : {out_path}")
    print(f"colonnes : {merged.columns.tolist()}")
    print("")

    print("--- comparaison sur les excursion connu ---")
    for device_id, (start, end) in excursion_windows.items():
        g = merged[merged["device_id"] == device_id]
        mask_window = (g["Time"] >= start) & (g["Time"] <= end)
        g_window = g[mask_window]

        print(f"device {device_id}")
        print(f"periode {start} a {end}")
        print(f"nb ligne {len(g_window)}")
        print(f"baseline flag {g_window['baseline_anomaly'].sum()}")
        print(f"zscore flag {g_window['zscore_anomaly'].sum()}")
        print(f"lstm flag {g_window['lstm_anomaly'].sum()}")
        print("")

    # comparaison precise : est ce que chaque methode voit LE pic exact
    # (fenetre etroite de +-30min) plutot que juste du bruit dans la semaine
    print("--- comparaison sur le pic exact (fenetre +-{}min) ---".format(PEAK_WINDOW_MINUTES))
    for device_id, peak_time_str in peak_timestamps.items():
        peak_time = pd.Timestamp(peak_time_str)
        window_start = peak_time - pd.Timedelta(minutes=PEAK_WINDOW_MINUTES)
        window_end = peak_time + pd.Timedelta(minutes=PEAK_WINDOW_MINUTES)

        g = merged[merged["device_id"] == device_id]
        mask_peak = (g["Time"] >= window_start) & (g["Time"] <= window_end)
        g_peak = g[mask_peak]

        print(f"device {device_id}")
        print(f"pic exact {peak_time_str}")
        print(f"fenetre {window_start} a {window_end}")
        print(f"nb ligne dans la fenetre {len(g_peak)}")
        print(f"baseline flag {g_peak['baseline_anomaly'].sum()} / {len(g_peak)}")
        print(f"zscore flag {g_peak['zscore_anomaly'].sum()} / {len(g_peak)}")
        print(f"lstm flag {g_peak['lstm_anomaly'].sum()} / {len(g_peak)}")

        # la temperature max observee dans la fenetre, pour confirmer
        # qu on est bien sur le bon pic
        if len(g_peak) > 0:
            print(f"temperature max dans la fenetre : {g_peak['Temperature'].max():.1f}")
        print("")

    # check sur TOUS les device, pas juste les 2 avec excursion connu
    # sert a verifier que le seuil par device (calibre sur 44B0C6 et 5F6A6A)
    # n explose pas les faux positif sur les device "sain"
    print("--- pourcentage de flag lstm par device (tous les device) ---")
    pct_par_device = (merged.groupby("device_id")["lstm_anomaly"].mean() * 100).round(2)
    is_device_avec_excursion = merged["device_id"].isin(excursion_windows.keys())
    resume = pd.DataFrame({
        "pourcentage_flag_lstm": pct_par_device,
    })
    resume["a_une_excursion_connu"] = resume.index.isin(excursion_windows.keys())
    resume = resume.sort_values("pourcentage_flag_lstm", ascending=False)
    print(resume)
    print("")

    # la plus longue serie de minute flag d affile, par device
    # un vrai incident donne un gros bloc continu, du bruit normal
    # donne juste des ligne flag isole ici et la
    print("--- plus longue serie de minute flag d affile (par device) ---")
    streaks = {}
    for device_id, group in merged.sort_values("Time").groupby("device_id"):
        streaks[device_id] = longest_flag_streak(group["lstm_anomaly"].values)

    streaks_df = pd.DataFrame({
        "plus_longue_serie_minutes": pd.Series(streaks),
    })
    streaks_df["a_une_excursion_connu"] = streaks_df.index.isin(excursion_windows.keys())
    streaks_df = streaks_df.sort_values("plus_longue_serie_minutes", ascending=False)
    print(streaks_df)
    print("")

    # regarde en detail ce qui se passe sur le device avec la plus longue serie
    print("--- detail de la plus longue serie (device avec le plus gros bloc) ---")
    device_a_regarder = streaks_df.index[0]
    print_streak_details(merged, device_a_regarder)

    print("evaluation terminee")


if __name__ == "__main__":
    main()