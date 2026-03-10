"""
Brain Tumor Detection – Streamlit Deployment (Bonus 10 Marks)
Upload an MRI image and get a prediction: Tumor / Normal
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "vgg16_brain_tumor.pth")
CLASS_NAMES = ["Normal", "Tumor"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.title("🧠 Brain Tumor Detection (VGG16)")
st.write("Upload an MRI brain image to classify it as **Tumor** or **Normal**.")

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    model = load_model()
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    st.markdown("---")
    if CLASS_NAMES[pred_idx] == "Tumor":
        st.error(f"**Prediction: {CLASS_NAMES[pred_idx]}**  (Confidence: {confidence:.1f}%)")
    else:
        st.success(f"**Prediction: {CLASS_NAMES[pred_idx]}**  (Confidence: {confidence:.1f}%)")

    st.bar_chart({"Normal": probs[0].item(), "Tumor": probs[1].item()})
