# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import ViTClassifier

# ==================================================
# CONFIG STREAMLIT
# ==================================================
st.set_page_config(
    page_title="Classification des feuilles de caféier Robusta",
    page_icon="🌿",
    layout="wide"
)

# ==================================================
# STYLE CSS
# ==================================================
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f0f8f5;
    color: #1f2933;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.author-card {
    background-color: #e8f5e9;
    border-left: 6px solid #4CAF50;
    color: #1f2933;
}

.card h3,
.card h4,
.card p,
.card b {
    color: #1f2933;
}

.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cpu")
CLASS_NAMES = ["Feuille saine", "Feuille malade"]

# ==================================================
# CHARGEMENT DU MODÈLE
# ==================================================
@st.cache_resource
def load_model():
    model = ViTClassifier(num_classes=2)
    model.load_state_dict(
        torch.load("best_vit_baseline.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==================================================
# TRANSFORMATIONS IMAGE
# ==================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# SIDEBAR
# ==================================================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📤 Charger image", "🔍 Prédiction", "👩‍🎓 Auteurs"]
)

# ==================================================
# ACCUEIL (CORRIGÉ)
# ==================================================
if menu == "🏠 Accueil":
    st.title("🌿 Classification intelligente des feuilles de caféier Robusta")

    st.markdown(
        "<h4 style='text-align:center; color:#4CAF50;'>"
        "Détection automatique de l'état de santé des feuilles (saines ou malade) par Vision Transformer"
        "</h4>",
        unsafe_allow_html=True
    )

    st.divider()
    st.subheader(" Contexte du projet")

    st.write(
        "La santé des plants de **caféier Robusta** est essentielle à la "
        "productivité agricole et à la qualité du produit final.\n\n"
        "Les **maladies foliaires** entraînent une baisse significative des "
        "rendements et compromettent la **durabilité des plantations**.\n\n"
        "🎯 **Objectif**  \n"
        "Développer un modèle basé sur un **Vision Transformer (ViT)** capable "
        "d’identifier automatiquement les **feuilles saines et malades** à partir "
        "d’images issues du jeu de données **RoCoLe**."
    )

# ==================================================
# CHARGEMENT IMAGE
# ==================================================
elif menu == "📤 Charger image":
    st.title("📤 Chargement de l’image")

    uploaded_file = st.file_uploader(
        "Formats acceptés : JPG, PNG, JPEG",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        st.session_state["uploaded_image"] = image

        if "history" not in st.session_state:
            st.session_state["history"] = []

        st.session_state["history"].append(image)

# ==================================================
# PRÉDICTION
# ==================================================
elif menu == "🔍 Prédiction":
    st.title("🔍 Détection de l'état de santé des feuilles")

    if "uploaded_image" not in st.session_state:
        st.info("Veuillez d’abord charger une image.")
    else:
        image = st.session_state["uploaded_image"]
        st.image(image, use_container_width=True)

        if st.button(" Lancer la prédiction"):
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)

            confidence, pred_class = torch.max(probs, dim=1)

            st.markdown("## 🧪 Résultat")

            if pred_class.item() == 1:
                st.error("❌ Feuille malade détectée")
            else:
                st.success("✅ Feuille saine détectée")

            st.metric("Probabilité", f"{confidence.item()*100:.2f}%")

            df_probs = pd.DataFrame({
                "Classe": CLASS_NAMES,
                "Probabilité": probs.cpu().numpy()[0]
            })

            st.bar_chart(df_probs.set_index("Classe"))

    if "history" in st.session_state:
        st.subheader("🖼️ Images analysées")
        cols = st.columns(4)
        for i, img in enumerate(st.session_state["history"][-4:]):
            cols[i % 4].image(img, use_container_width=True)

# ==================================================
# AUTEURS
# ==================================================
elif menu == "👩‍🎓 Auteurs":
    st.title(" Auteurs & Encadrement")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card author-card">
        <h3>Konaté Mariam</h3>
        <p><b>Spécialité :</b> Data Science / Data Analyst</p>
        <p><b>Université :</b> UFHB</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card author-card">
        <h3>Danho Amon Elisabeth Tania</h3>
        <p><b>Spécialité :</b> Data Science / Deep Learning</p>
        <p><b>Université :</b> UFHB</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b>Encadrant :</b> Dr Ayikpa<br>
    <b>Année académique :</b> 2025 – 2026
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("Application développée avec PyTorch & Streamlit")
