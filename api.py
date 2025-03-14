from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from PIL import Image

# Initialiser l'API
app = FastAPI()

# Charger le modèle
MODEL_PATH = "Deep_learning/models_saved/best_model.h5"

print("Chargement du modèle de deep learning...")
model = load_model(MODEL_PATH)
print("Modèle chargé avec succès.")

# Fonction de prétraitement de l'image
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # Adapter à la taille du modèle
    img_array = img_to_array(img) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    return img_array

# Endpoint pour prédire sur une image envoyée
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image, la prétraite et effectue une prédiction avec le modèle de deep learning.
    """
    try:
        # Charger l'image depuis le fichier uploadé
        image = Image.open(BytesIO(await file.read()))

        # Prétraitement de l'image
        img_array = preprocess_image(image)

        # Faire la prédiction
        res = model.predict(img_array)[0][0]

        # Interprétation du résultat
        diagnostic = "Positif" if res >= 0.5 else "Négatif"
        prob = res if res >= 0.5 else 1 - res

        return {
            "diagnostic": diagnostic,
            "probability": f"{prob:.2%}"
        }

    except Exception as e:
        return {"error": str(e)}
