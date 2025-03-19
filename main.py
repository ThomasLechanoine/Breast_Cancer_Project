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
from params import *
import tensorflow as tf

# ------------------------
# 📌 Imports du Deep Learning
from Deep_learning.dl_model import (
    create_feature_extractor,
    dl_initialize_edRVFL,
    dl_compile_model,
    dl_train_model
)
from Deep_learning.dl_custom_dataset import load_custom_dataset
from Deep_learning.dl_preprocess import extract_features




import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ---------------------
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
    print("🔄 Entraînement du modèle Machine Learning...")
    data = load_data(ML_DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, le = preprocess_data(data)
    best_model = tune_hyperparameters(X_train, y_train)

    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(best_model, ML_MODEL_PATH)
    joblib.dump(scaler, ML_SCALER_PATH)

    print("✅ Modèle ML entraîné et sauvegardé.")

# 📌 Entraîner le modèle DL
def train_dl():
    print("🔍 Vérification des données...")
    if not os.path.exists(DL_DATASET_PATH):
        print(f"❌ Erreur : Le dataset {DL_DATASET_PATH} n'existe pas.")
        return

    print("✅ Chargement des images avec CustomImageDataset...")
    train_ds, val_ds = load_custom_dataset(DL_DATASET_PATH, DL_IMG_SIZE, batch_size=16)

    print("✅ Extraction des caractéristiques avec VGG16...")
    feature_extractor = create_feature_extractor(DL_IMG_SIZE)

    print("🔄 Extraction des features pour train...")
    X_train_features, y_train = extract_features(feature_extractor, train_ds)
    print("🔄 Extraction des features pour validation...")
    X_val_features, y_val = extract_features(feature_extractor, val_ds)

    print(f"✅ Caractéristiques extraites : X_train {X_train_features.shape}, y_train {y_train.shape}")

    print("✅ Initialisation du modèle edRVFL...")
    input_dim = X_train_features.shape[1]
    model = dl_initialize_edRVFL(input_dim, num_classes=1, num_layers=5, hidden_units=50)

    print("🔄 Compilation du modèle...")
    model = dl_compile_model(model, optimizer=DL_OPTIMIZER, loss=DL_LOSS_FUNCTION, metrics=DL_METRICS)

    print("🚀 Entraînement du modèle...")
    model, history = dl_train_model(model, X_train_features, y_train, X_val_features, y_val, epochs=DL_EPOCHS, batch_size=32)

    print(f"🔄 Sauvegarde du modèle en cours dans : {DL_MODEL_PATH}")
    os.makedirs(os.path.dirname(DL_MODEL_PATH), exist_ok=True)

    try:
        model.save(DL_MODEL_PATH)
        print(f"✅ Modèle sauvegardé avec succès dans {DL_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du modèle : {e}")

# 📌 Point d'entrée principal
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", choices=["ml", "dl", "all"], help="Lancer l'entraînement des modèles")
    parser.add_argument("--evaluate_dl", action="store_true", help="Évaluer le modèle DL avec une matrice de confusion")
    parser.add_argument("--debug", action="store_true", help="Activer le mode debug")
    args = parser.parse_args()

    if args.debug:
        print("🔎 Mode DEBUG activé")

    if args.train == "ml":
        train_ml()
    elif args.train == "dl":
        train_dl()
    elif args.train == "all":
        train_ml()
        train_dl()

    # 📌 Nouvelle option pour évaluer le modèle DL
    if args.evaluate_dl:
        print("📊 Évaluation du modèle Deep Learning en cours...")
        os.system("python Deep_learning/dl_evaluate.py")  # Appelle le script d'évaluation
