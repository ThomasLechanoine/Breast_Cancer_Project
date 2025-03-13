import os  # Ajoute cette ligne en haut du fichier


GCP_PROJECT = os.environ.get("breast-cancer-project-453509")
GCP_REGION = os.environ.get("europe-west1-c")
BUCKET_NAME = os.environ.get("breast-cancer-bucket-images")
MODEL_TARGET = os.environ.get("local")
BUCKET_MODEL = os.environ.get("breast-cancer-bucket-model")
