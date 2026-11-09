from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification, logging
from PIL import Image
import torch

# Hide the messy warnings
logging.set_verbosity_error()

# Load the image
image_path = "/Users/mac/Desktop/test_image.jpg" 
image = Image.open(image_path).convert("RGB")

# Load processor and the correct V2 model (no mismatch bypass needed)
processor = MobileViTImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

# Process the image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Calculate probabilities and get the top 5 guesses
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
top5_prob, top5_indices = torch.topk(probabilities, 5)

print("\n--- Top 5 Predictions ---")
for i in range(5):
    class_name = model.config.id2label[top5_indices[i].item()]
    confidence = top5_prob[i].item() * 100
    print(f"{i+1}. {class_name}: {confidence:.2f}%")
