import os
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from Machine_learning.ml_preprocess import load_data, preprocess_data
from Machine_learning.ml_model import create_model, tune_hyperparameters, evaluate_model
from Deep_learning.dl_model import dl_initialize_model, dl_compile_model, dl_train_model
from Deep_learning.dl_preprocess import download
from params import *
from tensorflow.keras.preprocessing import image_dataset_from_directory


# 📌 Charger les modèles (évite le rechargement multiple)
def load_models():
    print("🔄 Chargement des modèles...")
    dl_model = load_model(DL_MODEL_PATH)
    ml_model = joblib.load(ML_MODEL_PATH)
    scaler = joblib.load(ML_SCALER_PATH)
    print("✅ Modèles chargés avec succès.")
    return dl_model, ml_model, scaler

# 📌 Prétraitement d'image pour le Deep Learning
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    return img_array

# 📌 Prétraitement des données tabulaires (Machine Learning)
def preprocess_ml_data(data_path):
    data = load_data(data_path)
    return preprocess_data(data)

# 📌 Entraîner le modèle ML
def train_ml():
    data = load_data(ML_DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, le = preprocess_data(data)
    best_model = tune_hyperparameters(X_train, y_train)

    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(best_model, ML_MODEL_PATH)
    joblib.dump(scaler, ML_SCALER_PATH)

    print("✅ Modèle ML entraîné et sauvegardé.")

# 📌 Entraîner le modèle DL
def train_dl():
    # Vérifier les données
    if not os.path.exists(DL_DATA_PATH):
        download()

    print("✅ Chargement des images...")

    # Charger les datasets
    train_dataset = image_dataset_from_directory(
        os.path.join(DL_DATA_PATH, "train"),
        labels="inferred",
        label_mode="binary",
        batch_size=DL_BATCH_SIZE,
        image_size=DL_IMG_SIZE,
        shuffle=True
    )

    valid_dataset = image_dataset_from_directory(
        os.path.join(DL_DATA_PATH, "valid"),
        labels="inferred",
        label_mode="binary",
        batch_size=DL_BATCH_SIZE,
        image_size=DL_IMG_SIZE,
        shuffle=True
    )

    # Initialisation et entraînement
    model = dl_initialize_model()
    model = dl_compile_model(model, optimizer=DL_OPTIMIZER, loss=DL_LOSS_FUNCTION, metrics=DL_METRICS)
    model, history = dl_train_model(model, train_dataset, valid_dataset, epochs=DL_EPOCHS)

    # Sauvegarde du modèle
    os.makedirs(os.path.dirname(DL_MODEL_PATH), exist_ok=True)
    try:
        print(f"🔄 Sauvegarde du modèle en cours dans : {DL_MODEL_PATH}")
        model.save(DL_MODEL_PATH)
        print(f"✅ Modèle sauvegardé avec succès dans {DL_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du modèle : {e}")

# 📌 Point d'entrée principal
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", choices=["ml", "dl", "all"], help="Lancer l'entraînement des modèles")
    args = parser.parse_args()

    if args.train == "ml":
        train_ml()
    elif args.train == "dl":
        train_dl()
    elif args.train == "all":
        train_ml()
        train_dl()
