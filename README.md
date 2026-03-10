# Brain Tumor Detection Using VGG16

A deep learning web application that classifies brain MRI images as **Tumor** or **Normal** using transfer learning with VGG16.

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## How It Works

1. Upload a brain MRI image (JPG/PNG)
2. The VGG16 model processes the image
3. Get an instant prediction with confidence score

## Tech Stack

- **PyTorch** – VGG16 transfer learning
- **Streamlit** – Interactive web interface
- **torchvision** – Image preprocessing & pretrained weights

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
