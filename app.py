import streamlit as st
import requests
from PIL import Image
import io

# Configuration de la page
st.set_page_config(page_title="Application de Détection de Cancer", layout="wide")

# Choix de la page
page = st.sidebar.selectbox("Choisissez une page", ["Prédiction Cancer (ML)","Prédiction Mammographie"])


# Page de prédiction via Machine Learning
if page == "Prédiction Cancer (ML)":
    st.title("Prédiction de Cancer via Machine Learning")

    st.write("Veuillez entrer les mesures de la tumeur pour obtenir une prédiction.")

    # Formulaire pour entrer les caractéristiques du modèle ML
    with st.form(key="prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            radius_mean = st.number_input("Radius Mean", min_value=0.0, value=17.99, format="%.4f")
            texture_mean = st.number_input("Texture Mean", min_value=0.0, value=10.38, format="%.4f")
            perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, value=122.8, format="%.4f")
            area_mean = st.number_input("Area Mean", min_value=0.0, value=1001.0, format="%.4f")
            smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, value=0.1184, format="%.4f")

        with col2:
            compactness_mean = st.number_input("Compactness Mean", min_value=0.0, value=0.2776, format="%.4f")
            concavity_mean = st.number_input("Concavity Mean", min_value=0.0, value=0.3001, format="%.4f")
            concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, value=0.1471, format="%.4f")
            symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, value=0.2419, format="%.4f")
            fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, value=0.07871, format="%.4f")

        with col3:
            radius_se = st.number_input("Radius SE", min_value=0.0, value=1.095, format="%.4f")
            texture_se = st.number_input("Texture SE", min_value=0.0, value=0.9053, format="%.4f")
            perimeter_se = st.number_input("Perimeter SE", min_value=0.0, value=8.589, format="%.4f")
            area_se = st.number_input("Area SE", min_value=0.0, value=153.4, format="%.4f")
            smoothness_se = st.number_input("Smoothness SE", min_value=0.0, value=0.006399, format="%.4f")

        st.write("### Mesures 'Worst' :")
        col4, col5, col6 = st.columns(3)

        with col4:
            radius_worst = st.number_input("Radius Worst", min_value=0.0, value=25.38, format="%.4f")
            texture_worst = st.number_input("Texture Worst", min_value=0.0, value=17.33, format="%.4f")
            perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, value=184.6, format="%.4f")
            area_worst = st.number_input("Area Worst", min_value=0.0, value=2019.0, format="%.4f")
            smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, value=0.1622, format="%.4f")

        with col5:
            compactness_worst = st.number_input("Compactness Worst", min_value=0.0, value=0.6656, format="%.4f")
            concavity_worst = st.number_input("Concavity Worst", min_value=0.0, value=0.7119, format="%.4f")
            concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, value=0.2654, format="%.4f")
            symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, value=0.4601, format="%.4f")
            fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, value=0.1189, format="%.4f")

        # ✅ Bouton de soumission correctement ajouté
        submit_button = st.form_submit_button(label="Lancer la Prédiction")

    # ✅ Vérification de la soumission
    if submit_button:
        api_url = "http://localhost:5000/predict_tumeur"  # Modifier avec l'URL de ton API

        # ✅ Préparer les données sous format JSON
        data = {
            "radius_mean": radius_mean,
            "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean,
            "area_mean": area_mean,
            "smoothness_mean": smoothness_mean,
            "compactness_mean": compactness_mean,
            "concavity_mean": concavity_mean,
            "concave_points_mean": concave_points_mean,
            "symmetry_mean": symmetry_mean,
            "fractal_dimension_mean": fractal_dimension_mean,
            "radius_se": radius_se,
            "texture_se": texture_se,
            "perimeter_se": perimeter_se,
            "area_se": area_se,
            "smoothness_se": smoothness_se,
            "radius_worst": radius_worst,
            "texture_worst": texture_worst,
            "perimeter_worst": perimeter_worst,
            "area_worst": area_worst,
            "smoothness_worst": smoothness_worst,
            "compactness_worst": compactness_worst,
            "concavity_worst": concavity_worst,
            "concave_points_worst": concave_points_worst,
            "symmetry_worst": symmetry_worst,
            "fractal_dimension_worst": fractal_dimension_worst
        }

        # ✅ Envoi de la requête à l'API
        response = requests.post(api_url, json=data)

        # ✅ Affichage du résultat
        if response.status_code == 200:
            result = response.json()
            st.success(f"Résultat : {result['prediction']}")
        else:
            st.error("Erreur lors de la requête à l'API")


if page == "Prédiction Mammographie":
    st.title("Prédiction de Cancer via Mammographie")

    st.write("Téléchargez une image de mammographie et appuyez sur **Prédiction** pour obtenir le résultat.")

    # ✅ Ajout d'un uploader pour charger une image
    uploaded_file = st.file_uploader("Téléchargez une image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Charger l'image avec PIL
        image = Image.open(uploaded_file)

        # ✅ Redimensionner l'image pour limiter sa taille (ex: 500px de large)
        max_width = 500  # Définir la largeur maximale
        w_percent = (max_width / float(image.size[0]))  # Calcul du ratio
        new_height = int((float(image.size[1]) * float(w_percent)))  # Calcul de la nouvelle hauteur
        image_resized = image.resize((max_width, new_height), Image.LANCZOS)

        # ✅ Affichage de l'image redimensionnée
        st.image(image_resized, caption="Image Originale (Redimensionnée)", use_container_width=True)

        # ✅ Convertir l'image en bytes pour l'envoyer à l'API
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # ✅ Bouton de prédiction
        if st.button("Lancer la prédiction"):
            api_url = "http://localhost:5000/predict_mammo"  # Modifier avec l'URL de ton API

            # Envoi de l'image à l'API
            files = {"file": ("image.png", img_bytes, "image/png")}
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                # Récupération des images renvoyées par l'API
                result = response.json()

                # Charger les images reçues
                original_image = Image.open(io.BytesIO(requests.get(result["original_image"]).content))
                predicted_image = Image.open(io.BytesIO(requests.get(result["predicted_image"]).content))

                # ✅ Redimensionner les images pour affichage
                original_image_resized = original_image.resize((max_width, new_height), Image.LANCZOS)
                predicted_image_resized = predicted_image.resize((max_width, new_height), Image.LANCZOS)

                # ✅ Ajout d'un slider pour naviguer entre les deux images
                image_choice = st.slider("Choisissez l'affichage", 1, 2, 1, format="Image %d")

                # Afficher l'image sélectionnée
                if image_choice == 1:
                    st.image(original_image_resized, caption="Image Originale", use_container_width=True)
                else:
                    st.image(predicted_image_resized, caption="Résultat du modèle", use_container_width=True)

            else:
                st.error("Erreur lors de la requête à l'API.")