import os
from PIL import Image
import streamlit as st
from transformers import CLIPModel, AutoProcessor, AutoTokenizer
import torch

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, tokenizer

model, processor, tokenizer = load_clip_model()

def load_images(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append((img, path))
        except Exception as e:
            st.warning(f"Failed to load image {path}: {e}")
    return images

def search_similar_images(text, images, threshold):
    text_inputs = tokenizer([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    results = []
    for img, path in images:
        image_inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
        score = similarity.item()
        if score >= threshold:
            results.append((img, path, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

st.title("CLIP based Image Search")
folder_path = st.text_input("Enter image folder path:", value="data/")

if folder_path and not os.path.isdir(folder_path):
    st.error("Path is not a valid folder.")

prompt = st.text_input("Enter search prompt:", value="Bus")
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.17, 0.01)

if st.button("Search"):
    if not folder_path or not os.path.isdir(folder_path):
        st.error("Enter a valid image folder.")
    elif not prompt:
        st.warning("Eenter a search prompt.")
    else:
        st.info("Searching...")
        images = load_images(folder_path)
        if not images:
            st.warning("No images found.")
        else:
            results = search_similar_images(prompt, images, threshold)

            if not results:
                st.warning("No matches found. Try reducing the threshold.")
            else:
                st.success(f"Found {len(results)} matches.")

                # cols = st.columns(3)
                cols = st.columns(4)
                for idx, (img, path, score) in enumerate(results):
                    # with cols[idx % 3]:
                    with cols[idx % 4]:
                        st.image(img, caption=f"{os.path.basename(path)}\nSimilarity: {score:.2f}", use_container_width=True)
