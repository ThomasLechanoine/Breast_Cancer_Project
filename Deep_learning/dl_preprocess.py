# prep

from google.colab import drive
import zipfile
#////////////////////////////////
def download():
    drive.mount('/content/drive')

    zip_path = "/content/drive/MyDrive/Data_prepros.zip"  # Modifier avec votre chemin réel

    #////////////////////////////////

    # Définir le chemin du fichier ZIP et le dossier de destination
    zip_path = "/content/drive/MyDrive/Data_prepros.zip"
    extract_path = "/content/data"  # Dossier où extraire les fichiers

    #////////////////////////////////
    # Extraire le fichier ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Extraction terminée !")

    #////////////////////////////////
    folder_path = "/content/data/Data_prepros"


    train_path = os.path.join(folder_path, "train")
    valid_path = os.path.join(folder_path, "valid")
    test_path = os.path.join(folder_path, "test")

#////////////////////////////////////////////////////////////////


#////////////////////////////////////////////////////////////////
