import pandas as pd

IN_PATH = "data/processed/all_data_with_baseline.csv"
OUT_PATH = "data/processed/all_data_with_zscore.csv"

# meme fenetre d excursion que dans baseline.py, sert juste pour comparer
# les 2 methode a la fin
excursion_windows = {
    "44B0C6": ("2026-06-20", "2026-06-27"),
    "5F6A6A": ("2026-07-13", "2026-07-19"),
}


# etape 1 : calcul le zscore glissant par device
# window=60 veut dire on regarde les 60 dernier minute pour la moyenne/std
# on fait ca par device separement sinon ca compare des device different
def apply_zscore(df, window=60, threshold=3):
    results = []

    for device_id, group in df.groupby("device_id"):
        g = group.sort_values("Time").copy()

        roll_mean = g["Temperature"].rolling(window, min_periods=10).mean()
        roll_std = g["Temperature"].rolling(window, min_periods=10).std()

        z = (g["Temperature"] - roll_mean) / roll_std
        g["zscore"] = z
        g["zscore_anomaly"] = z.abs() > threshold

        results.append(g)

    return pd.concat(results, ignore_index=True)


# etape 2 : compare baseline vs zscore sur les fenetre d excursion connu
# ca sert a voir si les 2 methode detecte bien les vrai excursion
def compare_on_excursions(df, excursion_windows):
    for device_id, (start, end) in excursion_windows.items():
        g = df[df["device_id"] == device_id]
        mask_window = (g["Time"] >= start) & (g["Time"] <= end)
        g_window = g[mask_window]

        print("device")
        print(device_id)
        print("periode")
        print(start, "a", end)
        print("nb ligne dans la fenetre")
        print(len(g_window))
        print("nb flag baseline dans la fenetre")
        print(g_window["baseline_anomaly"].sum())
        print("nb flag zscore dans la fenetre")
        print(g_window["zscore_anomaly"].sum())
        print("")


# programme principal
if __name__ == "__main__":

    print("chargement du fichier avec baseline deja fait")
    all_data = pd.read_csv(IN_PATH, parse_dates=["Time"])
    print("shape")
    print(all_data.shape)

    # window=30 choisi apres test (cf test_zscore_peak.py)
    # window plus grand (60+) rate le pic connu de 5F6A6A
    # window=30 detecte bien les 2 pic connu (44B0C6 et 5F6A6A)
    print("calcul du zscore glissant window 30 min threshold 3")
    all_data = apply_zscore(all_data, window=30, threshold=3)

    print("nb de ligne flag zscore_anomaly par device")
    print(all_data.groupby("device_id")["zscore_anomaly"].sum())

    print("pourcentage flag zscore par device")
    print((all_data.groupby("device_id")["zscore_anomaly"].mean() * 100).round(1))

    print("comparaison baseline vs zscore sur les excursion connu")
    compare_on_excursions(all_data, excursion_windows)

    print("sauvegarde du resultat")
    all_data.to_csv(OUT_PATH, index=False)
    print(OUT_PATH)