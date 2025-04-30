from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)

print(text_features.shape)

processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)

print(image_features.shape)

image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
logit_scale = model.logit_scale.exp()
torch.matmul(text_features, image_features.t()) * logit_scale

similarity = torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale
print(similarity)