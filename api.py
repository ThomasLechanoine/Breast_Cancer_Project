from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image
import joblib
from pydantic import BaseModel
from params import *

# Initialiser l'API
app = FastAPI()

# Charger le modèle DL
DL_MODEL_PATH = DL_MODEL_PATH
print("Chargement du modèle de deep learning...")
model = load_model(DL_MODEL_PATH)
print("✅ Modèle DL chargé avec succès.")

# Fonction de prétraitement de l'image
def preprocess_image(image_input):
    """
    Charge et prétraite une image depuis un fichier ou un objet BytesIO.
    """
    if isinstance(image_input, BytesIO):
        img = load_img(image_input, target_size=(224, 224))
    else:
        img = load_img(image_input, target_size=(224, 224))

    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Endpoint pour prédire sur une image envoyée
@app.post("/predict_dl")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire l'image et la convertir en BytesIO
        image_bytes = BytesIO(await file.read())

        # Prétraitement de l'image
        img_array = preprocess_image(image_bytes)

        # Vérifier si l'image a 1 canal (grayscale) et la convertir en RGB si nécessaire
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)

        # Faire la prédiction
        res = model.predict(img_array)[0][0]

        diagnostic = "Positif" if res >= 0.5 else "Négatif"
        prob = res if res >= 0.5 else 1 - res

        response = {"diagnostic": diagnostic, "probability": f"{prob:.2%}"}
        print("Réponse envoyée :", response)  # Debugging
        return response

    except Exception as e:
        print("Erreur API :", str(e))  # Debugging
        return {"error": str(e)}


# ------------------- Machine Learning Prediction -------------------

# Charger le modèle Machine Learning
ML_MODEL_PATH = ML_MODEL_PATH
SCALER_PATH = ML_SCALER_PATH

print("Chargement du modèle de Machine Learning...")
ml_model = joblib.load(ML_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Modèle de Machine Learning chargé avec succès.")

# Définir un schéma pour les données entrantes
class PredictionInput(BaseModel):
    features: list[float]

# Endpoint pour prédire sur des données tabulaires
@app.post("/predict_ml")
async def predict_ml(data: PredictionInput):
    """
    Reçoit une liste de caractéristiques, les prétraite et effectue une prédiction avec le modèle ML.
    """
    try:
        # Convertir en DataFrame avec les noms des colonnes attendus
        input_data = pd.DataFrame([data.features], columns=scaler.feature_names_in_)

        # Appliquer le scaling
        input_scaled = scaler.transform(input_data)

        # Faire la prédiction
        prediction = ml_model.predict(input_scaled)[0]
        diagnostic = "1= Malin (Cancer)" if prediction == 1 else "0= Bénin (Sans Cancer)"

        return {"diagnostic": diagnostic}

    except Exception as e:
        return {"error": str(e)}
