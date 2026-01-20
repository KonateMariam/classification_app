import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import ViTClassifier

# --------------------------------------------------
# CONFIG STREAMLIT
# --------------------------------------------------
st.set_page_config(
    page_title="Classification des feuilles de caféier Robusta",
    page_icon="🌿",
    layout="centered"
)

DEVICE = torch.device("cpu")  # Streamlit Cloud = CPU
CLASS_NAMES = ["Feuille saine", "Feuille malade"]

# --------------------------------------------------
# CHARGEMENT DU MODÈLE
# --------------------------------------------------
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

# --------------------------------------------------
# TRANSFORMATIONS IMAGE (IDENTIQUES AU TEST)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# INTERFACE UTILISATEUR
# --------------------------------------------------
st.title("🌿 Classification des feuilles de caféier Robusta")

st.markdown("""
### Contexte
La santé des plants de caféier Robusta est essentielle à la productivité agricole  
et à la durabilité des plantations.

Cette application repose sur un **Vision Transformer (ViT)** entraîné sur le  
jeu de données **RoCoLe**, afin d’identifier automatiquement :
- ✅ les feuilles **saines**
- ❌ les feuilles **malades**
""")

uploaded_file = st.file_uploader(
    "📤 Téléversez une image de feuille",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    st.subheader("🔍 Résultat de la prédiction")

    if pred_class.item() == 1:
        st.error(f"🌱 **Feuille malade**")
    else:
        st.success(f"🌿 **Feuille saine**")

    st.info(f"**Probabilité : {confidence.item() * 100:.2f} %**")
    st.progress(float(confidence))

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Projet académique – Classification des feuilles de caféier Robusta "
    "par Vision Transformer"
)
