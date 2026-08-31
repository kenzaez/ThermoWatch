
import os
import pandas as pd

FOLDER = "data/raw"
OUT_FOLDER = "data/processed"
os.makedirs(OUT_FOLDER, exist_ok=True)


def load_device(prefix, device_id, device_type, has_humidity=True, has_battery=True):
    df_temp = pd.read_csv(f"{FOLDER}/{prefix}_temp.csv")
    df_temp.columns = ["Time", "Temperature"]
    df_temp["Time"] = pd.to_datetime(df_temp["Time"])

    combined = df_temp

    if has_battery:
        df_battery = pd.read_csv(f"{FOLDER}/{prefix}_battery.csv")
        df_battery.columns = ["Time", "battery"]
        df_battery["Time"] = pd.to_datetime(df_battery["Time"])
        combined = combined.merge(df_battery, on="Time")

    if has_humidity:
        df_humidity = pd.read_csv(f"{FOLDER}/{prefix}_humidity.csv")
        df_humidity.columns = ["Time", "humidity"]
        df_humidity["Time"] = pd.to_datetime(df_humidity["Time"])
        combined = combined.merge(df_humidity, on="Time")

    combined["device_id"] = device_id
    combined["type"] = device_type
    return combined


def clean_and_resample(df, freq="1min", interp_limit=5):
    cleaned = []

    for device_id, group in df.groupby("device_id"):
        g = group.set_index("Time").sort_index()
        device_type = group["type"].iloc[0]

        value_cols = [c for c in ["Temperature", "humidity", "battery"] if c in g.columns]

        g_resampled = g[value_cols].resample(freq).mean()
        g_resampled = g_resampled.interpolate(limit=interp_limit)

        g_resampled["device_id"] = device_id
        g_resampled["type"] = device_type

        cleaned.append(g_resampled.reset_index())

    return pd.concat(cleaned, ignore_index=True)


def apply_baseline(df):
    df = df.copy()

    ranges = {
        "REF":   (2, 9),
        "CONG":  (-30, -20),
        "ETUVE": (35, 42),
    }

    def check_row(row):
        low, high = ranges[row["type"]]
        return not (low <= row["Temperature"] <= high)

    df["baseline_anomaly"] = df.apply(check_row, axis=1)
    return df

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

if __name__ == "__main__":


    devices = [
        load_device("CONG_5F6A6A", "5F6A6A", "CONG", has_humidity=False, has_battery=True),
        load_device("ARISTON_ CONG_D42730", "D42730", "CONG", has_humidity=False, has_battery=False),
        load_device("CANDYMINI_CONG_30ADB0", "30ADB0", "CONG", has_humidity=True, has_battery=True),
        load_device("BEKO_44B0C6", "44B0C6", "CONG", has_humidity=True, has_battery=True),
        load_device("HORECOLD_ REF_7E582A", "7E582A", "REF", has_humidity=False, has_battery=True),
        load_device("CANDYMINI_REF_8D07E0", "8D07E0", "REF", has_humidity=False, has_battery=True),
        load_device("ARISTON_ REF_A46F4D", "A46F4D", "REF", has_humidity=False, has_battery=False),
        load_device("REF_003FF8", "003FF8", "REF", has_humidity=True, has_battery=True),
        load_device("ETUVE_E9C2A6", "E9C2A6", "ETUVE", has_humidity=True, has_battery=True),
    ]

    all_data = pd.concat(devices, ignore_index=True)
    print("Raw combined:", all_data.shape)
    print(all_data["device_id"].value_counts())
    print(all_data.isnull().sum())

    all_data_clean = clean_and_resample(all_data)
    print("\nClean and resampled:", all_data_clean.shape)
    print(all_data_clean.isnull().sum())

    all_data_clean.to_csv(f"{OUT_FOLDER}/all_data_clean.csv", index=False)
    print(f"\nSaved: {OUT_FOLDER}/all_data_clean.csv")

    all_data_clean = apply_baseline(all_data_clean)
    print("\nBaseline anomalies total:", all_data_clean["baseline_anomaly"].sum())
    print(all_data_clean.groupby("device_id")["baseline_anomaly"].sum())

    all_data_clean = apply_zscore(all_data_clean)
    print("\nZ-score anomalies total:", all_data_clean["zscore_anomaly"].sum())
    print(all_data_clean.groupby("device_id")["zscore_anomaly"].sum())


    comparison = all_data_clean.groupby("device_id")[["baseline_anomaly", "zscore_anomaly"]].sum()
    print("\nComparaison baseline vs z-score:")
    print(comparison)

    all_data_clean.to_csv(f"{OUT_FOLDER}/all_data_with_anomalies.csv", index=False)
    print(f"\nSaved: {OUT_FOLDER}/all_data_with_anomalies.csv")