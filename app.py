import streamlit as st
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Triathlon Photo Local", page_icon="🚴", layout="centered")
st.title("🚴 Tri-Photo Clean Local (Illimité)")

url_input = st.text_input("Entrez l'URL de la photo (ou l'URL de la page) :")

def extract_image(page_url):
    """Récupère l'image depuis l'URL fournie."""
    headers = {"User-Agent": "Mozilla/5.0"}
    # Si l'utilisateur met directement le lien d'une image .jpg/.png
    if page_url.lower().endswith(('.png', '.jpg', '.jpeg')):
        res = requests.get(page_url, headers=headers)
        return Image.open(BytesIO(res.content))
    
    # Sinon, scraping de la page web
    try:
        res = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        meta_img = soup.find("meta", property="og:image")
        if meta_img:
            img_res = requests.get(meta_img["content"], headers=headers)
            return Image.open(BytesIO(img_res.content))
    except:
        pass
    return None

def remove_watermark_local(pil_image):
    """Efface les éléments clairs (filigranes fréquents) de l'image en local."""
    # Conversion PIL vers OpenCV (BGR)
    open_cv_image = np.array(pil_image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Création d'un masque automatique pour détecter le filigrane blanc/clair
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # Application de l'algorithme d'Inpainting (Gommage)
    dst = cv2.inpaint(open_cv_image, mask, 3, cv2.INPAINT_TELEA)
    
    # Reconversion vers PIL (RGB)
    return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

def upscale_local(pil_image):
    """Augmente artificiellement la netteté et la taille de l'image (X2)."""
    img = np.array(pil_image)
    # Redimensionnement bilinéaire simple mais ultra-rapide en local
    width = int(img.shape[1] * 2)
    height = int(img.shape[0] * 2)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(resized)

if url_input:
    if st.button("🚀 Nettoyer ma photo gratuitement"):
        with st.spinner("Téléchargement..."):
            img = extract_image(url_input)
            
        if img:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Originale")
                st.image(img, use_column_width=True)
                
            with col2:
                st.subheader("Traitée en local")
                with st.spinner("Calculs sur votre processeur..."):
                    # 1. On retire le filigrane
                    cleaned_img = remove_watermark_local(img)
                    # 2. On booste la qualité
                    final_img = upscale_local(cleaned_img)
                    
                st.image(final_img, use_column_width=True)
                
                # Téléchargement
                buf = BytesIO()
                final_img.save(buf, format="JPEG", quality=95)
                st.download_button("📥 Télécharger", data=buf.getvalue(), file_name="triathlon_local.jpg", mime="image/jpeg")
        else:
            st.error("Impossible de récupérer l'image. Inspectez la page et collez l'URL directe du .jpg")
