
import open_clip
from PIL import Image
import torch

# Load model with huggingface checkpoint
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:redlessone/DermLIP_ViT-B-16'
)
model.eval()

# Initialize tokenizer
tokenizer = open_clip.get_tokenizer('hf-hub:redlessone/DermLIP_ViT-B-16')

# Read example image
image = preprocess(Image.open("your_skin_image.png")).unsqueeze(0)

# Define disease labels (example: PAD dataset classes)
PAD_CLASSNAMES = [
    "nevus",
    "basal cell carcinoma",
    "actinic keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "melanoma"
]

# Build text prompts
template = lambda c: f'This is a skin image of {c}'
text = tokenizer([template(c) for c in PAD_CLASSNAMES])

# Inference
with torch.no_grad(), torch.autocast("cuda"):
    # Encode image and text
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Get prediction
final_prediction = PAD_CLASSNAMES[torch.argmax(text_probs[0])]
print(f'This image is diagnosed as {final_prediction}.')
print("Label probabilities:", text_probs)