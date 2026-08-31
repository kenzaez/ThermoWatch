from pathlib import Path

import numpy as np
import pandas as pd

DATA_FOLDER = Path(__file__).resolve().parent / "data" / "processed"
IN_PATH = DATA_FOLDER / "all_data_with_zscore.csv"
OUT_FOLDER = DATA_FOLDER

# meme fenetre d excursion connu que dans baseline.py et zscore.py
# on les enleve pour calculer la normalisation ET pour le training
excursion_windows = {
    "44B0C6": ("2026-06-20", "2026-06-27"),
    "5F6A6A": ("2026-07-13", "2026-07-19"),
}

# longueur de la sequence qu on donne au lstm (en minute)
SEQ_LENGTH = 60

# horizon = combien de minute a l avance on predit
# 1 = on predit juste la minute d apres (on pourra augmenter plus tard)
HORIZON = 1


# etape 1 : enleve les ligne qui sont dans une fenetre d excursion connu
# pour un device qui a pas d excursion connu on garde tout
def remove_excursions(df, excursion_windows):
    df = df.copy()
    mask_to_remove = pd.Series(False, index=df.index)

    for device_id, (start, end) in excursion_windows.items():
        is_this_device = df["device_id"] == device_id
        is_in_window = (df["Time"] >= start) & (df["Time"] <= end)
        mask_to_remove = mask_to_remove | (is_this_device & is_in_window)

    return df[~mask_to_remove].reset_index(drop=True)


# etape 2 : calcul mean/std par device sur les donnee SANS excursion
# ca sert de reference pour normaliser (z-score par device)
def get_device_norm_stats(df_no_excursion):
    stats = {}
    for device_id, group in df_no_excursion.groupby("device_id"):
        mean = group["Temperature"].mean()
        std = group["Temperature"].std()
        stats[device_id] = (mean, std)
    return stats


# etape 3 : applique la normalisation par device
# Temperature_norm = (Temperature - mean_device) / std_device
def apply_normalization(df, stats):
    df = df.copy()
    df["Temperature_norm"] = 0.0

    for device_id, (mean, std) in stats.items():
        mask = df["device_id"] == device_id
        df.loc[mask, "Temperature_norm"] = (df.loc[mask, "Temperature"] - mean) / std

    return df


# etape 4 : construit les sequence X et target y pour un seul device
# X = fenetre de SEQ_LENGTH minute, y = valeur HORIZON minute plus tard
# generique : si horizon=1 on est en mode 1-step ahead
# si horizon plus grand plus tard on pourra faire du multi-step
#
# rajoute : garde aussi le timestamp qui correspond a chaque y
# (le debut de la fenetre horizon), pour pouvoir recoller les
# prediction sur le csv original plus tard dans eval_lstm.py
def build_sequences(values, times, seq_length=SEQ_LENGTH, horizon=HORIZON):
    X = []
    y = []
    y_time = []

    for i in range(len(values) - seq_length - horizon + 1):
        seq_x = values[i : i + seq_length]
        seq_y = values[i + seq_length : i + seq_length + horizon]
        X.append(seq_x)
        y.append(seq_y)
        # timestamp de la premiere valeur qu on essaie de predire
        y_time.append(times[i + seq_length])

    return np.array(X), np.array(y), np.array(y_time)


# etape 5 : pour un type de device (REF/CONG/ETUVE), construit toute
# les sequence en concatenant les devices de ce type
# important : on construit les sequence PAR DEVICE separement puis on
# concatene apres, sinon une sequence pourrait melanger 2 device different
#
# rajoute : garde aussi time_final et device_final, alignee un a un
# avec y_final, pour le recollage cote eval_lstm.py
def build_sequences_for_type(df, device_type, seq_length=SEQ_LENGTH, horizon=HORIZON):
    df_type = df[df["type"] == device_type].sort_values(["device_id", "Time"])

    all_X = []
    all_y = []
    all_time = []
    all_device = []

    for device_id, group in df_type.groupby("device_id"):
        values = group["Temperature_norm"].values
        times = group["Time"].values

        X, y, y_time = build_sequences(values, times, seq_length, horizon)

        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_time.append(y_time)
            # meme device_id repete pour chaque sequence de ce device
            all_device.append(np.full(len(X), device_id))

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    time_final = np.concatenate(all_time, axis=0)
    device_final = np.concatenate(all_device, axis=0)

    return X_final, y_final, time_final, device_final


# programme principal
if __name__ == "__main__":

    print("chargement du fichier avec zscore deja fait")
    all_data = pd.read_csv(IN_PATH, parse_dates=["Time"])
    print("shape")
    print(all_data.shape)

    print("suppression des ligne d excursion connu pour la normalisation et le training")
    df_no_excursion = remove_excursions(all_data, excursion_windows)
    print("shape apres suppression excursion")
    print(df_no_excursion.shape)

    print("calcul mean et std par device sur donnee sans excursion")
    device_stats = get_device_norm_stats(df_no_excursion)
    for device_id, (mean, std) in device_stats.items():
        print(device_id)
        print("mean", round(mean, 2), "std", round(std, 2))

    print("application de la normalisation z-score par device")
    df_no_excursion = apply_normalization(df_no_excursion, device_stats)

    print("construction des sequence par type de device")
    print("seq length", SEQ_LENGTH, "horizon", HORIZON)

    for device_type in df_no_excursion["type"].unique():
        X, y, y_time, y_device = build_sequences_for_type(df_no_excursion, device_type)

        print("type")
        print(device_type)
        print("shape X")
        print(X.shape)
        print("shape y")
        print(y.shape)

        # sauvegarde en npy, plus simple que csv pour des tableau numpy
        np.save(f"{OUT_FOLDER}/X_{device_type}.npy", X)
        np.save(f"{OUT_FOLDER}/y_{device_type}.npy", y)
        # nouveau : timestamp et device_id de chaque sequence
        # allow_pickle pas necessaire pour time (datetime64) mais device_id
        # est du texte donc numpy le sauvegarde en object, ca marche pareil
        np.save(f"{OUT_FOLDER}/time_{device_type}.npy", y_time)
        np.save(f"{OUT_FOLDER}/device_{device_type}.npy", y_device)
        print("sauvegarde fait")
        print(f"{OUT_FOLDER}/X_{device_type}.npy")
        print(f"{OUT_FOLDER}/y_{device_type}.npy")
        print(f"{OUT_FOLDER}/time_{device_type}.npy")
        print(f"{OUT_FOLDER}/device_{device_type}.npy")
        print("")

    print("resume")
    print("devices utilise pour la normalisation (mean/std)")
    print(list(device_stats.keys()))
    print("lignes d excursion connu retire du training")
    print(len(all_data) - len(df_no_excursion))