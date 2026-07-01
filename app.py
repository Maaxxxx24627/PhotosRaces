import streamlit as st
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Triathlon Photo Local", page_icon="🚴", layout="centered")
st.title("🚴 Tri-Photo Clean Local (Version Stable)")

# --- INTERFACE D'IMPORTATION ---
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
            except: return None
        try:
            res = requests.get(page_url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            meta_img = soup.find("meta", property="og:image")
            if meta_img:
                img_res = requests.get(meta_img["content"], headers=headers, timeout=10)
                return Image.open(BytesIO(img_res.content))
        except: pass
        return None

    if url_input:
        with st.spinner("Téléchargement de l'image..."):
            img_source = extract_image(url_input)
else:
    uploaded_file = st.file_uploader("Glissez votre photo ici", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_source = Image.open(uploaded_file)

# --- TRAITEMENT ET REGLAGES ---
if img_source:
    st.success("Image chargée avec succès !")
    
    # Préparation des images au format OpenCV
    img_np = np.array(img_source)
    if img_np.shape[-1] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    st.subheader("🎛️ Réglages du filtre de détection")
    st.write("Ajustez les curseurs pour cibler le texte du filigrane sans toucher au reste de la photo.")

    # Curseurs pour ajuster précisément la détection du filigrane transparent
    sensibilite = st.slider("Seuil de détection du blanc (Plus bas = détecte plus de transparence)", 150, 255, 200)
    epaisseur = st.slider("Élargissement de la zone de gommage (Taille des lettres)", 1, 10, 3)

    # Création du masque basé sur l'ajustement de l'utilisateur
    _, mask = cv2.threshold(gray, sensibilite, 255, cv2.THRESH_BINARY)
    
    # Dilatation du masque pour englober les bords des lettres texturées
    kernel = np.ones((epaisseur, epaisseur), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Affichage du masque pour que l'utilisateur comprenne ce qui va être effacé (les zones blanches)
    st.write("👁️ **Zone détectée (en blanc) :**")
    st.image(mask, caption="Le texte du filigrane doit apparaître clairement en blanc ici", use_container_width=True)

    # --- BOUTON DE LANCEMENT ---
    if st.button("🚀 Lancer la reconstruction de la photo et l'Upscale"):
        with st.spinner("L'algorithme reconstruit les textures cachées sous le filigrane..."):
            # 1. Inpainting mathématique basé sur le masque ajusté
            dst = cv2.inpaint(img_cv, mask, 5, cv2.INPAINT_TELEA)
            cleaned_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

            # 2. Super-Résolution locale (X2) pour contrer la perte de qualité
            orig_w, orig_h = img_source.size
            width = int(orig_w * 2)
            height = int(orig_h * 2)
            final_cv = cv2.resize(cleaned_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
            final_img = Image.fromarray(final_cv)

        # Affichage du résultat final
        st.subheader("✨ Votre Photo Propre et Boostée")
        st.image(final_img, use_container_width=True)

        # Préparation du téléchargement
        buf = BytesIO()
        final_img.save(buf, format="JPEG", quality=98)
        st.download_button(
            label="📥 Télécharger mon image HD",
            data=buf.getvalue(),
            file_name="triathlon_hd_propre.jpg",
            mime="image/jpeg"
        )
