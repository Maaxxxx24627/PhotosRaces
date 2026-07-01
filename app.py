import streamlit as st
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Triathlon Photo Local", page_icon="🚴", layout="centered")
st.title("🚴 Tri-Photo Clean Local (Illimité)")

# --- ZONE DE SÉLECTION DE LA SOURCE ---
st.write("Choisissez votre méthode pour importer la photo de triathlon :")
source_option = st.radio("Méthode d'importation :", ("Coller un lien / URL", "Charger un fichier depuis mon PC"), horizontal=True)

img_source = None

if source_option == "Coller un lien / URL":
    url_input = st.text_input("Entrez l'URL de la photo ou de la page web :")
    
    def extract_image(page_url):
        headers = {"User-Agent": "Mozilla/5.0"}
        if page_url.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                res = requests.get(page_url, headers=headers, timeout=10)
                return Image.open(BytesIO(res.content))
            except:
                return None
        try:
            res = requests.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            meta_img = soup.find("meta", property="og:image")
            if meta_img:
                img_res = requests.get(meta_img["content"], headers=headers, timeout=10)
                return Image.open(BytesIO(img_res.content))
        except:
            pass
        return None

    if url_input:
        with st.spinner("Téléchargement de l'image depuis le web..."):
            img_source = extract_image(url_input)
        if img_source is None:
            st.error("Impossible de récupérer l'image via ce lien. Essayez d'inspecter la page pour trouver le .jpg direct, ou téléchargez-la d'abord sur votre PC.")

else:
    uploaded_file = st.file_uploader("Glissez votre photo ici ou cliquez pour parcourir", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_source = Image.open(uploaded_file)

# --- FONCTIONS DE TRAITEMENT LOCAL ---
def remove_watermark_local(pil_image):
    """Efface les éléments clairs (filigranes fréquents) de l'image en local."""
    open_cv_image = np.array(pil_image)
    # Gestion des images PNG avec transparence
    if open_cv_image.shape[2] == 4:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Détection des pixels très clairs (filigranes blancs/gris clair)
    _, mask = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    # Inpainting algorithmique (Navier-Stokes)
    dst = cv2.inpaint(open_cv_image, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

def upscale_local(pil_image):
    """Augmente la taille (X2) et améliore la netteté en local."""
    img = np.array(pil_image)
    width = int(img.shape[1] * 2)
    height = int(img.shape[0] * 2)
    # Interpolation Lanczos4 pour préserver les bords et détails nets
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(resized)

# --- ZONE DE TRAITEMENT ET AFFICHAGE ---
if img_source:
    st.success("Image chargée avec succès !")
    
    if st.button("🚀 Nettoyer et Améliorer ma photo"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Originale")
            st.image(img_source, use_container_width=True)
            
        with col2:
            st.subheader("Résultat Local HD")
            with st.spinner("Calculs en cours sur votre processeur..."):
                # 1. Suppression filigrane
                cleaned_img = remove_watermark_local(img_source)
                # 2. Amélioration résolution
                final_img = upscale_local(cleaned_img)
                
            st.image(final_img, use_container_width=True)
            
            # Préparation du fichier de téléchargement
            buf = BytesIO()
            final_img.save(buf, format="JPEG", quality=95)
            
            st.download_button(
                label="📥 Télécharger la photo propre",
                data=buf.getvalue(),
                file_name="triathlon_clean.jpg",
                mime="image/jpeg"
            )
