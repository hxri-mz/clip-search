import os
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def load_images_from_folder(folder_path):
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path)
                   if fname.lower().endswith(supported_exts)]
    images = []
    for path in image_paths:
        try:
            images.append((Image.open(path).convert("RGB"), path))
        except Exception as e:
            print(f"Could not load image {path}: {e}")
    return images

def find_best_match(text, image_folder):
    text_inputs = tokenizer([text], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    images = load_images_from_folder(image_folder)
    if not images:
        print("No valid images found.")
        return

    similarities = []
    image_paths = []
    
    for img, path in images:
        image_inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
        similarities.append(similarity.item())
        image_paths.append(path)

    best_idx = torch.tensor(similarities).argmax().item()
    print(f"\nBest match: {image_paths[best_idx]}")
    print(f"Similarity Score: {similarities[best_idx]:.4f}")

    return image_paths[best_idx]

if __name__ == "__main__":
    folder = "/home/minus/sim2real/embsimilarity/data/"
    query_text = "excavator"
    find_best_match(query_text, folder)
