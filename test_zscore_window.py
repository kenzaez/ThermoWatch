import pandas as pd

IN_PATH = "data/processed/all_data_with_baseline.csv"

# le moment EXACT du pic connu pour chaque device (trouve dans l EDA)
# pas toute la semaine, juste la minute du vrai pic
known_peaks = {
    "44B0C6": "2026-06-23 12:45",
    "5F6A6A": "2026-07-16 14:02",
}

# marge autour du pic, pour pas etre trop strict sur la minute exacte
margin_minutes = 5

# les valeur de window qu on veut tester
windows_to_test = [15, 30, 60, 120, 180, 360]

THRESHOLD = 3


# meme fonction que dans zscore.py
def apply_zscore(df, window, threshold):
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


# regarde si le pic exact (+/- margin_minutes) est flag ou pas
def check_peak_flagged(df, device_id, peak_time, margin_minutes):
    g = df[df["device_id"] == device_id]
    peak_time = pd.to_datetime(peak_time)

    start = peak_time - pd.Timedelta(minutes=margin_minutes)
    end = peak_time + pd.Timedelta(minutes=margin_minutes)

    mask = (g["Time"] >= start) & (g["Time"] <= end)
    g_around_peak = g[mask]

    nb_ligne = len(g_around_peak)
    nb_flag = g_around_peak["zscore_anomaly"].sum()
    peak_est_flag = nb_flag > 0

    return nb_ligne, nb_flag, peak_est_flag


# programme principal
if __name__ == "__main__":

    print("chargement du fichier avec baseline deja fait")
    all_data = pd.read_csv(IN_PATH, parse_dates=["Time"])
    print("shape")
    print(all_data.shape)

    print("test si le pic exact est flag, pour chaque window")
    print("marge utilisee autour du pic en minute")
    print(margin_minutes)
    print("")

    for window in windows_to_test:
        print("=====================================")
        print("window testee")
        print(window)

        all_data_z = apply_zscore(all_data, window=window, threshold=THRESHOLD)

        for device_id, peak_time in known_peaks.items():
            nb_ligne, nb_flag, peak_est_flag = check_peak_flagged(
                all_data_z, device_id, peak_time, margin_minutes
            )
            print("device")
            print(device_id)
            print("nb ligne autour du pic")
            print(nb_ligne)
            print("nb flag autour du pic")
            print(nb_flag)
            print("pic detecte")
            print(peak_est_flag)

        print("")