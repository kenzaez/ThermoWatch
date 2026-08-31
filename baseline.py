import pandas as pd

IN_PATH = "data/processed/all_data_clean.csv"
OUT_PATH = "data/processed/all_data_with_baseline.csv"

# fenetre de temps ou on sait qu il y a une vrai excursion
# on les enleve avant de calculer les seuils sinon ca fausse le percentile
excursion_windows = {
    "44B0C6": ("2026-06-20", "2026-06-27"),
    "5F6A6A": ("2026-07-13", "2026-07-19"),
}


# etape 1 : calcul le seuil bas/haut par device avec percentile 5 et 95
# on enleve les excursion connu avant de calculer sinon le seuil sera fausse
def get_clean_thresholds(df, excursion_windows, low_q=0.05, high_q=0.95):
    thresholds = {}

    for device_id, group in df.groupby("device_id"):
        g = group.copy()

        # si ce device a une fenetre d excursion connu on l enleve du calcul
        if device_id in excursion_windows:
            start, end = excursion_windows[device_id]
            g = g[~((g["Time"] >= start) & (g["Time"] <= end))]

        low = g["Temperature"].quantile(low_q)
        high = g["Temperature"].quantile(high_q)
        thresholds[device_id] = (low, high)

    return thresholds


# etape 2 : applique les seuils par device pour flag les anomalie
# chaque device a son propre seuil, pas un seuil commun par type
# (decision prise apres l EDA, les device du meme type sont trop different)
def apply_baseline_per_device(df, thresholds):
    df = df.copy()
    df["baseline_anomaly"] = False

    for device_id, (low, high) in thresholds.items():
        mask = df["device_id"] == device_id
        df.loc[mask, "baseline_anomaly"] = ~df.loc[mask, "Temperature"].between(low, high)

    return df


# programme principal
if __name__ == "__main__":

    print("chargement du fichier clean")
    all_data_clean = pd.read_csv(IN_PATH, parse_dates=["Time"])
    print("shape")
    print(all_data_clean.shape)

    print("calcul des seuil par device percentile 5 et 95 excursion exclu")
    device_thresholds = get_clean_thresholds(all_data_clean, excursion_windows)
    print("seuils trouve par device")
    for device_id, (low, high) in device_thresholds.items():
        print(device_id)
        print(round(low, 1), "to", round(high, 1))

    print("application du baseline par device")
    all_data_clean = apply_baseline_per_device(all_data_clean, device_thresholds)

    print("nb de ligne flag anomaly par device")
    print(all_data_clean.groupby("device_id")["baseline_anomaly"].sum())

    print("pourcentage flag par device")
    print((all_data_clean.groupby("device_id")["baseline_anomaly"].mean() * 100).round(1))

    print("sauvegarde du resultat")
    all_data_clean.to_csv(OUT_PATH, index=False)
    print(OUT_PATH)