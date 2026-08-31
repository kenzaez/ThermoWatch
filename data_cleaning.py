import os
import pandas as pd

# dossier ou sont les fichier brut
FOLDER = "data/raw"
# dossier pour mettre le resultat
OUT_FOLDER = "data/processed"
os.makedirs(OUT_FOLDER, exist_ok=True)


# fonction pour charger un device (meme chose que dans le fichier exploration)
def load_device(prefix, device_id, device_type, has_humidity=True, has_battery=True):
    # on lit le csv de temperature
    df_temp = pd.read_csv(f"{FOLDER}/{prefix}_temp.csv")
    df_temp.columns = ["Time", "Temperature"]
    df_temp["Time"] = pd.to_datetime(df_temp["Time"])

    combined = df_temp

    # si le device a un capteur batterie on ajoute
    if has_battery:
        df_battery = pd.read_csv(f"{FOLDER}/{prefix}_battery.csv")
        df_battery.columns = ["Time", "battery"]
        df_battery["Time"] = pd.to_datetime(df_battery["Time"])
        combined = combined.merge(df_battery, on="Time", how="outer")

    # si le device a un capteur humidite on ajoute
    if has_humidity:
        df_humidity = pd.read_csv(f"{FOLDER}/{prefix}_humidity.csv")
        df_humidity.columns = ["Time", "humidity"]
        df_humidity["Time"] = pd.to_datetime(df_humidity["Time"])
        combined = combined.merge(df_humidity, on="Time", how="outer")

    # on rajoute les colone info device
    combined["device_id"] = device_id
    combined["type"] = device_type
    return combined.sort_values("Time").reset_index(drop=True)


# liste de tt les devices avec leur info (prefix fichier, id, type, a humidite?, a battery?)
DEVICES_CONFIG = [
    ("CONG_5F6A6A", "5F6A6A", "CONG", False, True),
    ("ARISTON_ CONG_D42730", "D42730", "CONG", False, False),
    ("CANDYMINI_CONG_30ADB0", "30ADB0", "CONG", True, True),
    ("BEKO_44B0C6", "44B0C6", "CONG", True, True),
    ("HORECOLD_ REF_7E582A", "7E582A", "REF", False, True),
    ("CANDYMINI_REF_8D07E0", "8D07E0", "REF", False, True),
    ("ARISTON_ REF_A46F4D", "A46F4D", "REF", False, False),
    ("REF_003FF8", "003FF8", "REF", True, True),
    ("ETUVE_E9C2A6", "E9C2A6", "ETUVE", True, True),
]


# etape 1 : on met chaque device sur une grille de temps regulier (1min)
# et on comble les petit trou avec interpolation
def clean_and_resample(df, freq="1min", interp_limit=5):
    # ici on fait pour chaque device separement, sinon ca marche pas bien
    cleaned = []

    for device_id, group in df.groupby("device_id"):
        g = group.set_index("Time").sort_index()
        device_type = group["type"].iloc[0]

        # on garde que les colone qui existe pour ce device
        value_cols = [c for c in ["Temperature", "humidity", "battery"] if c in g.columns]
        g_resampled = g[value_cols].resample(freq).mean()

        # on interpole SEULEMENT temperature
        # humidity/battery manquant = pas de capteur, pas un trou a combler
        # (si on interpole ca va inventer des valeur fausse)
        if "Temperature" in g_resampled.columns:
            g_resampled["Temperature"] = g_resampled["Temperature"].interpolate(limit=interp_limit)

        g_resampled["device_id"] = device_id
        g_resampled["type"] = device_type
        cleaned.append(g_resampled.reset_index())

    return pd.concat(cleaned, ignore_index=True)


# etape 2 : calcul le changement de temperature minute par minute
def compute_temp_diff(df):
    df = df.sort_values("Time").copy()
    # tres important : on fait groupby device_id sinon ca compare 2 device
    # different entre eux (bug qu on avait trouve avant)
    df["temp_diff"] = df.groupby("device_id")["Temperature"].diff().abs()
    return df


# etape 3 : detecter les segment bizzare (plateau plat trop long)
def flag_flatline_segments(df, min_flat_minutes=15, flat_tolerance=0.01):
    # ca sert a trouver les endroit ou la courbe est toute plate pendant
    # longtemps, ca sent le fake (genre grafana qui a rempli avec ligne droite
    # pendant une panne du capteur)
    # on efface rien, on met juste un flag pour etre honnete dans le rapport
    df = df.sort_values(["device_id", "Time"]).copy()
    df["suspicious_flat"] = False

    for device_id, group in df.groupby("device_id"):
        idx = group.index
        is_flat = group["temp_diff"].fillna(0) <= flat_tolerance

        # compte les serie qui se suivent de is_flat
        run_id = (is_flat != is_flat.shift()).cumsum()
        run_lengths = is_flat.groupby(run_id).transform("sum")

        suspicious = is_flat & (run_lengths >= min_flat_minutes)
        df.loc[idx[suspicious.values], "suspicious_flat"] = True

    return df


# etape 4 : on ajoute un groupe selon les capteur que le device a
# groupe A = juste temp, groupe B = temp+battery, groupe C = temp+humidity+battery
# comme ca on evite d avoir des nan tout le temps pour certain device
def add_sensor_group(df):
    df = df.copy()

    # on fait un dico id -> (has_humidity, has_battery) a partir du config
    sensor_info = {dev_id: (has_h, has_b) for _, dev_id, _, has_h, has_b in DEVICES_CONFIG}

    def get_group(device_id):
        has_h, has_b = sensor_info[device_id]
        if has_h and has_b:
            return "C"
        elif has_b and not has_h:
            return "B"
        else:
            return "A"

    df["sensor_group"] = df["device_id"].apply(get_group)
    return df


# etape 5 : separe le dataframe en 3 dataframe selon le sensor_group
# ca sert pour apres, pour faire un model par groupe (pas de nan a gerer)
def split_by_sensor_group(df):
    groups = {}
    for g, sub in df.groupby("sensor_group"):
        # on enleve les colone qui servent a rien pour ce groupe (tt nan)
        cols_to_keep = ["Time", "Temperature", "device_id", "type", "temp_diff", "suspicious_flat"]
        if g == "B":
            cols_to_keep.append("battery")
        if g == "C":
            cols_to_keep.append("battery")
            cols_to_keep.append("humidity")

        cols_present = [c for c in cols_to_keep if c in sub.columns]
        groups[g] = sub[cols_present].reset_index(drop=True)

    return groups


# programme principal
if __name__ == "__main__":

    print("chargement des 9 devices")
    devices = [load_device(prefix, dev_id, dev_type, has_h, has_b)
               for prefix, dev_id, dev_type, has_h, has_b in DEVICES_CONFIG]

    all_data = pd.concat(devices, ignore_index=True)
    print("shape brute")
    print(all_data.shape)

    print("resample et interpolation limit 5 min")
    all_data_clean = clean_and_resample(all_data, freq="1min", interp_limit=5)
    print("shape apres resample")
    print(all_data_clean.shape)
    print(all_data_clean.isnull().sum())

    print("calcul temp diff par device")
    all_data_clean = compute_temp_diff(all_data_clean)
    print(all_data_clean.groupby("type")["temp_diff"].agg(["mean", "median", "max"]))

    print("detection des segment suspect")
    all_data_clean = flag_flatline_segments(all_data_clean)
    suspicious_summary = all_data_clean.groupby("device_id")["suspicious_flat"].sum()
    print("nb de ligne marque suspicious_flat par device")
    print(suspicious_summary)

    print("ajout du sensor group A B ou C selon les capteur du device")
    all_data_clean = add_sensor_group(all_data_clean)
    print(all_data_clean.groupby("sensor_group")["device_id"].unique())

    # sauvegarde du resultat
    out_path = f"{OUT_FOLDER}/all_data_clean.csv"
    all_data_clean.to_csv(out_path, index=False)
    print("sauvegarde fait")
    print(out_path)

    print("separation en 3 dataframe selon sensor group pour le modeling")
    grouped = split_by_sensor_group(all_data_clean)
    for g, sub_df in grouped.items():
        group_path = f"{OUT_FOLDER}/group_{g}.csv"
        sub_df.to_csv(group_path, index=False)
        print("groupe")
        print(g)
        print("shape")
        print(sub_df.shape)
        print("sauvegarde fait")
        print(group_path)

    # petit resume pour le rapport
    print("resume pour le rapport")
    print("lignes totales apres nettoyage")
    print(len(all_data_clean))
    print("devices sans humidite")
    print([d for _, d, _, h, _ in DEVICES_CONFIG if not h])
    print("devices sans batterie")
    print([d for _, d, _, _, b in DEVICES_CONFIG if not b])
    print("interp limit utilise 5 minutes taux de variation moyen environ 0.075 degre par min")
    print("lignes totales marque comme segment suspect")
    print(all_data_clean['suspicious_flat'].sum())