import streamlit as st
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Triathlon Photo Local", page_icon="🚴", layout="centered")
st.title("🚴 Tri-Photo Clean Local (Illimité)")

# --- SOURCE SELECTION ---
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

# --- CONFIGURATION DE LA BROSSE ---
if img_source:
    st.success("Image chargée ! Utilisez la souris pour colorier les filigranes ci-dessous.")
    
    # Ajustement de la taille de l'affichage pour le dessin
    orig_w, orig_h = img_source.size
    max_width = 600
    display_w = max_width
    display_h = int(orig_h * (max_width / orig_w))
    img_resized = img_source.resize((display_w, display_h))

    stroke_width = st.slider("Taille de la brosse d'effacement :", 5, 50, 15)

    st.write("✏️ **Coloriez grossièrement les écritures 'PHOTO Running' :**")
    
    # Zone de dessin interactive
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Couleur orange transparente pour voir où on dessine
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_image=img_resized,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="freedraw",
        key="canvas",
    )

    # --- TRAITEMENT LOCAL ---
    if st.button("🚀 Effacer le filigrane dessiné et Upscaler"):
        if canvas_result.image_data is not None:
            with st.spinner("Calculs de reconstruction en cours..."):
                # 1. Préparation du masque créé par l'utilisateur
                mask = canvas_result.image_data[:, :, 3] # Récupère la couche alpha du dessin
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

                # 2. Conversion de l'image d'origine pour OpenCV
                img_cv = np.array(img_source)
                if img_cv.shape[2] == 4:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

                # 3. Application de l'Inpainting algorithmique ciblé
                # Augmenter le rayon (ex: 7) aide à combler les filigranes épais
                dst = cv2.inpaint(img_cv, mask, 7, cv2.INPAINT_TELEA)
                cleaned_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

                # 4. Upscaling local (LANCZOS4 pour préserver les textures de la trifonction et du visage)
                width = int(orig_w * 2)
                height = int(orig_h * 2)
                final_cv = cv2.resize(cleaned_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
                final_img = Image.fromarray(final_cv)

            # Affichage du résultat final
            st.subheader("✨ Résultat Nettoyé et Boosté")
            st.image(final_img, use_container_width=True)

            # Bouton de téléchargement
            buf = BytesIO()
            final_img.save(buf, format="JPEG", quality=98)
            st.download_button(
                label="📥 Télécharger ma photo HD propre",
                data=buf.getvalue(),
                file_name="triathlon_resultat.jpg",
                mime="image/jpeg"
            )
        else:
            st.warning("Veuillez d'abord dessiner sur le filigrane avant de lancer le traitement.")
