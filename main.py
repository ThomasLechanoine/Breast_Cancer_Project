
# ////////////////////////// IMPORT //////////////////////////

import numpy as np
from params import *
import joblib

# // IMPORT ML///
from Machine_learning.ml_preprocess import load_data, preprocess_data
from Machine_learning.ml_model import create_model, tune_hyperparameters, evaluate_model

# // IMPORT DL///
import os
import numpy as np
import tensorflow as tf
from Deep_learning.dl_model import dl_initialize_model, dl_compile_model, dl_train_model
from Deep_learning.dl_preprocess import download  # Si besoin de télécharger et prétraiter les données

#prediction DL
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
from Deep_learning.dl_preprocess import preprocess_image  # Importer la fonction de preprocessing

# ///////////////////// VISUALISATIONS //////////////////////



# //////////////////// MACHINE LEARNING /////////////////////
"""
fonctions for machine learning based on CSV
"""

# Charger et prétraiter les données
data = load_data("Data/Machine_learning.csv")  # Modifier avec le bon chemin
X_train, X_test, y_train, y_test, scaler, le = preprocess_data(data)

# Initialiser et optimiser le modèle
model = create_model()
best_model = tune_hyperparameters(X_train, y_train)

# Évaluer le modèle
evaluate_model(best_model, X_test, y_test)


# ///////////////////// DEEP LEARNING ////////////////////
"""
fonctions for deep learning based on CSV
"""

# /// PARAMÈTRES ///
DATA_DIR = "Data/Data_Deep_Learning/"  # Modifier si le dossier des images est ailleurs
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

# Vérifier si les données existent, sinon les télécharger
if not os.path.exists(DATA_DIR):
    print("Téléchargement et extraction des données...")
    download()

# /// CHARGEMENT DES DONNÉES //
print("Chargement des images...")

train_dataset = image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="binary",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

valid_dataset = image_dataset_from_directory(
    os.path.join(DATA_DIR, "valid"),
    labels="inferred",
    label_mode="binary",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

# /// INITIALISATION DU MODÈLE ///
print("Initialisation du modèle...")
model = dl_initialize_model()

# /// COMPILATION DU MODÈLE ///
print("Compilation du modèle...")
model = dl_compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall'])

# /// ENTRAÎNEMENT DU MODÈLE ///
print("Entraînement du modèle...")
model, history = dl_train_model(model, train_dataset, valid_dataset, epochs=EPOCHS)

# // SAUVEGARDE DU MODÈLE //
print("Sauvegarde du modèle entraîné...")
# Définition du chemin de sauvegarde
MODEL_SAVE_PATH = "Deep_learning/models_saved/best_model.h5"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Sauvegarde du modèle
print(f"Sauvegarde du modèle dans {MODEL_SAVE_PATH} ...")
model.save(MODEL_SAVE_PATH)

print(f"✅ Modèle sauvegardé avec succès dans {MODEL_SAVE_PATH} !")

print("Entraînement terminé et modèle sauvegardé sous 'best_model.h5' !")


# ///////////////////// PREDICTION_ML ////////////////////

def ml_predict(input_data):
    """
    Fonction pour faire une prédiction avec le modèle de Machine Learning.
    """
    # Charger le modèle ML et le scaler
    model_path = "Machine_learning/models_saved/ml_best_model.pkl"
    scaler_path = "Machine_learning/models_saved/ml_scaler.pkl"


    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Normaliser les données
    input_data_scaled = scaler.transform([input_data])

    # Faire la prédiction
    prediction = model.predict(input_data_scaled)[0]
    diagnostic = "Malin (Cancer)" if prediction == 1 else "Bénin (Sans Cancer)"

    return diagnostic

# Test de prédiction ML
demo_data = X_test[0]  # Exemple avec un élément du jeu de test
result = ml_predict(demo_data)
print(f"Prédiction Machine Learning : {result}")


# ///////////////////// PREDICTION_DL ////////////////////

MODEL_PATH = "/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/models_saved/best_model.h5"


def load_trained_model():
    """
    Charge le modèle entraîné depuis le fichier best_model.h5.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le modèle {MODEL_PATH} n'existe pas. Entraîne-le d'abord.")

    print("Chargement du modèle entraîné...")
    model = load_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès.")
    return model

def predictImage(image_path, model):
    '''Takes an image and a model
    '''
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = img_array.reshape((-1, 150, 150, 3))
    res = model.predict(img_array)[0][0]

    if res < 0.5:
        diagnostic = "Négatif"
        prob = 1 - res
    else:
        diagnostic = "Positif"
        prob = res

    plt.imshow(array_to_img(img_array[0]))
    plt.axis("off")
    plt.title(f"{diagnostic} ({prob:.2%})")

    return  plt.show()

# ///////////////////// END ////////////////////
