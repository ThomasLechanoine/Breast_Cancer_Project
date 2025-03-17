import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Charger le modèle
MODEL_PATH = "Deep_learning/models_saved/best_model.h5"
print("🔄 Chargement du modèle...")
model = load_model(MODEL_PATH)
print("✅ Modèle chargé avec succès.")

# Fonction de prétraitement de l'image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adapter à la taille du modèle
    img_array = img_to_array(img) / 255.0  # Normalisation
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    return img_array

# Liste des images à tester
images = {
    "Positive (Cancer)": os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "Data", "Data_Deep_Learning", "test", "1", "1025_773597682_png.rf.10eef608772845290c0ed1c5dc80c3ac.jpg"),
    "Négative (Pas de cancer)": os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "Data", "Data_Deep_Learning", "test", "0", "30_275846607_png.rf.3137e89295140dd04a7e09885bd8c9b6.jpg")
}

# Tester chaque image
for label, image_path in images.items():
    try:
        print(f"\n🖼️ Chargement de l'image : {label}")
        img_array = preprocess_image(image_path)

        # Faire la prédiction
        res = model.predict(img_array)[0][0]

        # Log des valeurs brutes
        print(f"🔍 Valeur brute de la prédiction : {res:.4f}")

        # Interprétation
        threshold = 0.4
        diagnostic = "Positif" if res >= threshold else "Négatif"
        prob = res if res >= threshold else 1 - res

        print(f"🩺 Diagnostic : {diagnostic} ({prob:.2%})")

    except Exception as e:
        print(f"❌ Erreur lors de la prédiction pour {label} : {str(e)}")


import os
pos_images = len(os.listdir("/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/Data/Data_Deep_Learning/train/1"))
neg_images = len(os.listdir("/home/bren/code/ThomasLechanoine/Breast_Cancer_Project/Data/Data_Deep_Learning/train/0"))

print(f"📊 Nombre d'images positives : {pos_images}")
print(f"📊 Nombre d'images négatives : {neg_images}")
