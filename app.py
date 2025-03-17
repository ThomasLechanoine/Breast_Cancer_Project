import streamlit as st
import requests
from PIL import Image
import io
import tensorflow as tf
import joblib
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from params import *  # Importation des URLs API

# Machine Learning Imports
from Machine_learning.ml_preprocess import load_data, preprocess_data
from Machine_learning.ml_model import create_model, tune_hyperparameters, evaluate_model

# Add Machine_learning/ to sys.path for module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Machine_learning")))

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(page_title="Application de Détection de Cancer du sein", layout="wide")

# Image à afficher à gauche dans la sidebar
image_path_left = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img", "01.png") #<------------------------------------------------
image = Image.open(image_path_left)

# Afficher l'image sur la barre latérale
st.sidebar.image(image_path_left, use_container_width=True)

# Load and display the cover image
# image_path = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img", "01.png")
# st.image(image_path, use_container_width=True)

# ---------------------- CUSTOM PAGE NAVIGATION ----------------------
st.sidebar.title("Navigation")

# Maintain session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Graphiques"

if st.sidebar.button("Graphiques"):
    st.session_state.page = "Graphiques"
if st.sidebar.button("Prédiction Mammographie (DL)"):
    st.session_state.page = "Prédiction Mammographie (DL)"
if st.sidebar.button("Prédiction Cancer (ML)"):
    st.session_state.page = "Prédiction Cancer (ML)"

page = st.session_state.page


# ---------------------- GRAPHICS PAGE ----------------------
if page == "Graphiques":
    st.title("Visualisation des Graphiques")
    st.write("Analyse des données avec des visualisations graphiques.")

    # Définition du répertoire contenant les graphiques
    graph_dir = os.path.join("/home", "bren", "code", "ThomasLechanoine", "Breast_Cancer_Project", "app_img") #<------------------------------------------------

    # Liste des graphiques avec descriptions
    graph_data = [
        {"file": "image_graph1.png", "title": "Graphique 1", "description": "📊 Ce graphique montre la distribution des caractéristiques du dataset."},
        {"file": "image_graph2.png", "title": "Graphique 2", "description": "🔬 Cette visualisation met en évidence la corrélation entre les différentes variables."},
        {"file": "image_graph3.png", "title": "Graphique 3", "description": "📈 Analyse des performances du modèle avec différentes métriques d’évaluation."}
    ]

    # Affichage des images avec menu déroulant pour description
    for graph in graph_data:
        img_path = os.path.join(graph_dir, graph["file"])

        with st.expander(f"📊 {graph['title']}"):
            st.image(img_path, use_column_width=True)
            st.write(graph["description"])

# Ajout de style CSS pour rendre le contour du menu déroulant plus visible
st.markdown("""
    <style>
        /* Style pour rendre le contour du menu déroulant plus visible */
        div[data-testid="stExpander"] {
            border: 2px solid #4A90E2 !important; /* Bleu vif */
            border-radius: 10px !important;
            background-color: #E3F2FD !important; /* Bleu pastel */
            padding: 10px !important;
        }

        /* Style du titre dans l'expander */
        div[data-testid="stExpander"] summary {
            font-weight: bold !important;
            font-size: 16px !important;
            color: #1A1A1A !important;
        }
    </style>
""", unsafe_allow_html=True)


# ---------------------- LOAD MODELS DL---------------------
@st.cache_resource
def load_dl_model():
    return tf.keras.models.load_model(DL_MODEL_PATH)
 #//////////

model = load_dl_model()

# ---------------------- LOAD MODELS ML---------------------
# Load the trained model and scaler
@st.cache_resource
def load_model():
    MODEL_PATH = ML_MODEL_PATH #<------------------------------------------------
    SCALER_PATH = ML_SCALER_PATH #<------------------------------------------------
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

# ---------------------- LOAD test data--------------------- a voir si besoin
@st.cache_resource
def load_test_data():
    dataset_path = ML_DATA_PATH #<------------------------------------------------
    data = pd.read_csv(dataset_path)
    X = data.drop(columns=["id", "diagnosis"])  # Drop unnecessary columns
    y = data["diagnosis"].map({"B": 0, "M": 1})  # Encode labels (B:0, M:1)

    # Split into train and test (must match how the model was trained)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_test, y_test

# Load test dataset
X_test, y_test = load_test_data()


#//////////////////////Page de prediction DEEP LEARNING/////////////////////////////

if page == "Prédiction Mammographie (DL)":
    # Configuration de la page
    st.title("Prédiction de Cancer via Deep Learning")

        # Ajout du sous-titre et explication du Deep Learning
    st.subheader("Qu'est-ce que le Deep Learning ?")

    with st.expander("Définition du Deep Learning (Expliqué simplement)"):
        st.write("""
        🔍 Le **Deep Learning** est une branche de l'intelligence artificielle.

        🔍 Imagine un enfant qui apprend à reconnaître un chat en voyant beaucoup d'images de chats.
            Le Deep Learning fait pareil !
            Avec des **milliers d'exemples**, il devient de plus en plus fort pour **reconnaître** des objets, des visages, des animaux, etc.

        🔍 **Exemple** : Un modèle de Deep Learning peut analyser une mammographie et dire si une tumeur est présente ou non.
        """)

    # Ajout d'un deuxième sous-titre avant l'input d'image
    st.subheader("Analyse de mammographie")

    st.write("Téléchargez une image de mammographie et appuyez sur **Prédiction** pour obtenir le résultat.")

    # Ajout d'un uploader pour charger une image
    uploaded_file = st.file_uploader("Téléchargez une image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    DL_API_URL = DL_API_URL  # Plus tard, il suffira de changer cette URL avec l' API cloud #<------------------------------------------------

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)

        # Resize image to fit in a smaller area (e.g., max width 500px)
        max_width = 500  # Set maximum width
        w_percent = max_width / float(image.size[0])
        new_height = int(float(image.size[1]) * w_percent)  # Maintain aspect ratio
        image_resized = image.resize((max_width, new_height), Image.LANCZOS)

        # Display resized image
        st.image(image_resized, caption="Image Redimensionnée", use_container_width=False)

        # Bouton de prédiction
        if st.button("Lancer la prédiction"):
            # Convertir l'image en bytes pour l'envoyer à l'API
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            # Envoi de l'image à l'API
            files = {"file": ("image.png", img_bytes, "image/png")}
            response = requests.post(DL_API_URL, files=files)

            # Vérification de la réponse
            if response.status_code == 200:
                result = response.json()
                st.success(f"Résultat : {result['diagnostic']} ({result['probability']})")
            else:
                st.error("Erreur lors de la requête à l'API.")



# ///////////Page de prédiction via Machine Learning////////////
#----------------------------------------------
st.markdown("""
    <style>
        /* Fond général en bleu pastel */
        .stApp {
            background-color: #E3F2FD !important;
        }

        /* Titres en couleur foncée pour contraste */
        h1, h2, h3, h4, h5, h6, p, label {
            color: #1A1A1A !important;
            font-weight: bold !important;
        }

        /* Bouton principal (st.button et st.form_submit_button) */
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background-color: #FFA69E !important; /* Rouge saumon pastel */
            color: #FFFFFF !important; /* Texte blanc */
            border-radius: 12px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 12px 24px !important;
            border: none !important; /* Suppression des bordures */
            box-shadow: none !important; /* Suppression de l'ombre */
            transition: all 0.3s ease-in-out !important;
        }

        /* Effet hover sur les boutons */
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #FF6B6B !important; /* Rouge plus foncé */
            box-shadow: none !important; /* Suppression de l’ombre */
            transform: scale(1.05) !important; /* Effet léger d'agrandissement */
        }

        /* Style des inputs */
        .stNumberInput>div>div {
            width: 140px !important;
            border-radius: 8px !important;
            border: 2px solid #A1C4FD !important; /* Bleu pastel */
            background-color: #FFFFFF !important;
            color: #1A1A1A !important;
            padding: 5px !important;
        }

        /* Texte des inputs */
        div[data-testid="stNumberInput"] input {
            font-size: 14px !important;
            padding: 10px !important;
            text-align: center !important;
            background-color: #FFFFFF !important;
            color: #1A1A1A !important;
            border: none !important;
        }

        /* Amélioration des boutons + / - uniquement sur la page Machine Learning */
        div[data-testid="stNumberInput"] button {
            background-color: #4A90E2 !important; /* Bleu pastel */
            color: #FFFFFF !important; /* Texte blanc */
            border-radius: 6px !important;
            font-size: 14px !important;
            font-weight: bold !important;
            padding: 6px 12px !important;
            border: none !important;
            transition: all 0.2s ease-in-out !important;
        }

        /* Effet hover sur les boutons + / - */
        div[data-testid="stNumberInput"] button:hover {
            background-color: #357ABD !important; /* Bleu plus foncé */
            transform: scale(1.1) !important; /* Effet léger d'agrandissement */
        }

        /* Sections dépliables (Expander) */
        .st-expander {
            background-color: #B6D0E2 !important;
            border: 2px solid #7DA0B6 !important;
            color: #1A1A1A !important;
        }

        /* Sidebar */
        .stSidebar {
            background-color: #B2D3FF !important;
        }

        /* Style des résultats de prédiction */
        .stSuccess {
            border-radius: 10px !important;
            padding: 10px !important;
            text-align: center !important;
            font-weight: bold !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------- Machine Learning Prediction -------------------
# URL de l'API pour la prédiction ML
ML_API_URL = ML_API_URL  # !!! Remplacer cette URL si l'API est hébergée en ligne #<------------------------------------------------


# ------------------- Machine Learning Prediction -------------------
if page == "Prédiction Cancer (ML)":
    st.title("Prédiction de Cancer via Machine Learning")


    #  Valeurs par défaut (corrigées)
    default_values_1 = {
        "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0,
        "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001, "concave points_mean": 0.1471,
        "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871, "radius_se": 1.095, "texture_se": 0.9053,
        "perimeter_se": 8.589, "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904,
        "concavity_se": 0.05373, "concave points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
        "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
        "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119, "concave points_worst": 0.2654,
        "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
    }

    default_values_2 = {
        "radius_mean": 13.64, "texture_mean": 16.34, "perimeter_mean": 87.21, "area_mean": 571.8,
        "smoothness_mean": 0.07685, "compactness_mean": 0.06059, "concavity_mean": 0.01857, "concave points_mean": 0.01723,
        "symmetry_mean": 0.1353, "fractal_dimension_mean": 0.05953, "radius_se": 0.1872, "texture_se": 0.9234,
        "perimeter_se": 1.449, "area_se": 14.55, "smoothness_se": 0.004477, "compactness_se": 0.01177,
        "concavity_se": 0.01079, "concave points_se": 0.007956, "symmetry_se": 0.01325, "fractal_dimension_se": 0.002551,
        "radius_worst": 14.67, "texture_worst": 23.19, "perimeter_worst": 96.08, "area_worst": 656.7,
        "smoothness_worst": 0.1089, "compactness_worst": 0.1582, "concavity_worst": 0.105, "concave points_worst": 0.08586,
        "symmetry_worst": 0.2346, "fractal_dimension_worst": 0.08025
    }

    # ------------------- PRÉDICTION 1 -------------------


    # Ajout du sous-titre et explication du Machine Learning
    st.subheader("Qu'est-ce que le Machine Learning ?")

    with st.expander("Définition du Machine Learning (Expliqué simplement)"):
        st.write("""
        🔍 **Le Machine Learning (ML)** est une branche de l'intelligence artificielle.

        🎯 Plutôt que d’être **programmés manuellement** pour chaque tâche, les modèles de Machine Learning trouvent **eux-mêmes des paterns** dans les données.

        🏥 **Exemple médical** : En analysant des **milliers de tumeurs**, un modèle peut **prédire** si une nouvelle tumeur est bénigne ou maligne, simplement en comparant ses caractéristiques avec celles de tumeurs déjà connues.
        """)

    # Ajout d'un deuxième sous-titre avant l'input des caractéristiques tumorales
    st.subheader("Analyse des caractéristiques de la tumeur")
    st.write("Veuillez entrer les mesures de la tumeur pour obtenir une prédiction.")

    st.subheader("Prédiction 1 (Maligne)")
    with st.form(key="prediction_form_1"):
        columns = st.columns(5)
        feature_values_1 = {}

        for i, feature in enumerate(default_values_1.keys()):
            with columns[i % 5]:
                feature_values_1[feature] = st.number_input(
                    feature, min_value=0.0, format="%.4f", value=default_values_1[feature]
                )

        submit_button_1 = st.form_submit_button(label="Lancer la Prédiction 1")


    if submit_button_1:
        input_data_1 = pd.DataFrame([list(feature_values_1.values())], columns=default_values_1.keys())

        if input_data_1.isnull().values.any():
            st.error("⚠️ Certaines valeurs sont vides ou incorrectes ! Veuillez remplir tous les champs.")
        else:
            input_data_json = {"features": input_data_1.values.tolist()[0]}
            response = requests.post(ML_API_URL, json=input_data_json)

            if response.status_code == 200:
                prediction_1 = response.json()["diagnostic"]
            else:
                prediction_1 = "Erreur lors de la prédiction."

            # **Mise en forme du résultat**
            if "Malin" in prediction_1:
                diagnostic_1 = "🔴 Malin (Cancer)"
                color_1 = "#F76C6C"  # Rouge pastel
            else:
                diagnostic_1 = "🔵 Bénin (Sans Cancer)"
                color_1 = "#A1C4FD"  # Bleu pastel

            st.markdown(
                f'<div style="background-color:{color_1}; padding:15px; border-radius:10px; text-align:center; '
                f'font-size:16px; color:white; font-weight:bold;">'
                f'Résultat de la prédiction 1 : {diagnostic_1}'
                '</div>',
                unsafe_allow_html=True
            )
    # ------------------- PRÉDICTION 2 -------------------
    st.subheader("Prédiction 2 (Bénigne)")
    with st.form(key="prediction_form_2"):
        columns = st.columns(5)
        feature_values_2 = {}

        for i, feature in enumerate(default_values_2.keys()):
            with columns[i % 5]:
                feature_values_2[feature] = st.number_input(
                    feature, min_value=0.0, format="%.4f", value=default_values_2[feature]
                )

        submit_button_2 = st.form_submit_button(label="Lancer la Prédiction 2")

    if submit_button_2:
        input_data_2 = pd.DataFrame([list(feature_values_2.values())], columns=default_values_2.keys())

        if input_data_2.isnull().values.any():
            st.error("⚠️ Certaines valeurs sont vides ou incorrectes ! Veuillez remplir tous les champs.")
        else:
            input_data_json = {"features": input_data_2.values.tolist()[0]}
            response = requests.post(ML_API_URL, json=input_data_json)

            if response.status_code == 200:
                prediction_2 = response.json()["diagnostic"]
            else:
                prediction_2 = "Erreur lors de la prédiction."

            #  **Mise en forme du résultat**
            if "Malin" in prediction_2:
                diagnostic_2 = "🔴 Malin (Cancer)"
                color_2 = "#F76C6C"  # Rouge pastel
            else:
                diagnostic_2 = "🔵 Bénin (Sans Cancer)"
                color_2 = "#A1C4FD"  # Bleu pastel

            st.markdown(
                f'<div style="background-color:{color_2}; padding:15px; border-radius:10px; text-align:center; '
                f'font-size:16px; color:white; font-weight:bold;">'
                f'Résultat de la prédiction 2 : {diagnostic_2}'
                '</div>',
                unsafe_allow_html=True
            )

   #--------------------CONFUSION MATRIX------------------
    if submit_button_1 or submit_button_2:
        # Select appropriate input
        input_data = input_data_1 if submit_button_1 else input_data_2

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_data_scaled)[0]
        diagnostic = "Malin (Cancer)" if prediction == 1 else "Bénin (Sans Cancer)"

        # Display the result
        st.success(f"Résultat de la prédiction : {diagnostic}")

        # Compute Confusion Matrix
        y_pred = model.predict(scaler.transform(X_test))
        cm = confusion_matrix(y_test, y_pred)

        # Custom colormap similar to provided UI colors
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#e1f0ff", "#a7c8f2", "#6da0e5", "#2e75c5"])

        # Display the Confusion Matrix as a Heatmap with customized colors
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='g', cmap=custom_cmap, cbar=True,
                    linewidths=1, linecolor='white', ax=ax)
        ax.set_xlabel("Valeurs Prédites", fontsize=12, color='#333333')
        ax.set_ylabel("Valeurs Réelles", fontsize=12, color='#333333')
        ax.set_title("Matrice de Confusion", fontsize=14, color='#333333')

        # Improve tick labels for visibility
        ax.xaxis.set_tick_params(labelsize=10, colors='#333333')
        ax.yaxis.set_tick_params(labelsize=12, colors='#333333')

        plt.tight_layout()
        st.pyplot(fig)
