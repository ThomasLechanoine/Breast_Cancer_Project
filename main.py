
# ////////////////////////// IMPORT //////////////////////////

import numpy as np

# from Deep_learning import ()
# from Machine_learning import ()

from params import *

# ///////////////////// VISUALISATIONS //////////////////////



# //////////////////// MACHINE LEARNING /////////////////////
"""
fonctions for machine learning based on CSV
"""

# def ml_preprocess_and_train()

    # Preprocess data using ml_preprocess.py


# ///////////////////// DEEP LEARNING ////////////////////
"""
fonctions for deep learning based on CSV
"""

# // IMPORT ///
import os
import numpy as np
import tensorflow as tf
from Deep_learning.dl_model import dl_initialize_model, dl_compile_model, dl_train_model
from Deep_learning.dl_preprocess import download  # Si besoin de télécharger et prétraiter les données
from tensorflow.keras.preprocessing import image_dataset_from_directory

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
MODEL_SAVE_PATH = "/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/models_saved/best_model.h5"

# Sauvegarde du modèle
print(f"Sauvegarde du modèle dans {MODEL_SAVE_PATH} ...")
model.save(MODEL_SAVE_PATH)

print(f"✅ Modèle sauvegardé avec succès dans {MODEL_SAVE_PATH} !")


print("Entraînement terminé et modèle sauvegardé sous 'best_model.h5' !")




# ///////////////////// PREDICTION_ML ////////////////////

# def pred():
#     model = load_model()

#     print(f"✅ pred() done")

#     return

# ///////////////////// PREDICTION_DL ////////////////////
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from Deep_learning.dl_preprocess import preprocess_image  # Importer la fonction de preprocessing

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

# def predictImage(image_path):
#     """
#     Prend un chemin d'image et un modèle, et renvoie le résultat de la prédiction.
#     """
#     model = load_trained_model()
#     img_array = preprocess_image(image_path)
#     res = model.predict(img_array)[0][0]

#     diagnostic = "Positif" if res >= 0.5 else "Négatif"
#     prob = res if res >= 0.5 else 1 - res

#     plt.imshow(array_to_img(img_array[0]))
#     plt.axis("off")
#     plt.title(f"{diagnostic} ({prob:.2%})")
#     plt.show()

#     return diagnostic, prob  # Retourne aussi les valeurs pour affichage dans Streamlit

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