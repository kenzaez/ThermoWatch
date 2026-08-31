import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

DATA_FOLDER = "data/processed"
MODEL_FOLDER = "models"

import os
os.makedirs(MODEL_FOLDER, exist_ok=True)

# les 3 type de device, chacun aura son propre model
# ETUVE en premier car c est le plus petit dataset, pour tester vite la vitesse
TYPES = ["ETUVE", "REF", "CONG"]

# pourcentage des donnee utilise pour validation (le reste = training)
# on prend les DERNIERE donnee pour validation, pas random
# car ici c est du temporel, on veut valider sur du "futur" pas du passe
VAL_SPLIT = 0.2

EPOCHS = 10
BATCH_SIZE = 64


# etape 1 : charge X et y pour un type, et les mets a la bonne forme
# keras LSTM veut la forme (nb exemple, seq_length, nb feature)
# nous on a 1 seule feature (Temperature_norm), donc on rajoute cette dimension
def load_data(device_type):
    X = np.load(f"{DATA_FOLDER}/X_{device_type}.npy")
    y = np.load(f"{DATA_FOLDER}/y_{device_type}.npy")

    # X etait (nb exemple, seq_length), on rajoute une dimension a la fin
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


# etape 2 : split en train / validation
# PAS de shuffle ici, on garde l ordre, sinon on "trichera" en entrainant
# sur des donnee qui viennent temporellement apres celle de validation
def train_val_split(X, y, val_split=VAL_SPLIT):
    split_idx = int(len(X) * (1 - val_split))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    return X_train, y_train, X_val, y_val


# etape 3 : construit le model lstm
# archi simple : 1 couche LSTM qui lit la sequence, puis 1 couche Dense
# qui sort une seule valeur (la prediction de la minute suivante)
def build_model(seq_length):
    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        # 32 = nb de "neurone" dans la couche lstm, plus y en a plus le model
        # peut apprendre de pattern complexe, mais plus lent a entrainer
        layers.LSTM(32),
        # derniere couche, sort 1 seule valeur = la temperature normalise predite
        layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")
    return model


# programme principal
if __name__ == "__main__":

    for device_type in TYPES:
        print("=====================================")
        print("entrainement pour le type")
        print(device_type)

        print("chargement des donnee")
        X, y = load_data(device_type)
        print("shape X")
        print(X.shape)
        print("shape y")
        print(y.shape)

        print("split train/val (derniere partie = validation)")
        X_train, y_train, X_val, y_val = train_val_split(X, y)
        print("nb exemple train")
        print(len(X_train))
        print("nb exemple validation")
        print(len(X_val))

        print("construction du model")
        seq_length = X.shape[1]
        model = build_model(seq_length)
        model.summary()

        print("debut de l entrainement")
        # epoch = 1 passage complet sur toute les donnee de training
        # batch_size = combien d exemple le model regarde avant de s ajuster
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

        print("entrainement fini pour")
        print(device_type)
        print("derniere loss training")
        print(history.history["loss"][-1])
        print("derniere loss validation")
        print(history.history["val_loss"][-1])

        # sauvegarde du model entraine, pour pouvoir le reutiliser apres
        # sans avoir a re entrainer depuis zero
        model_path = f"{MODEL_FOLDER}/lstm_{device_type}.keras"
        model.save(model_path)
        print("model sauvegarde")
        print(model_path)
        print("")