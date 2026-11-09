import gradio as gr
from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification, logging
import torch
from PIL import Image

# Hide warnings
logging.set_verbosity_error()

# Load processor and model once at startup
processor = MobileViTImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

def classify_image(image):
    # Process the uploaded image
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    outputs = model(**inputs)
    
    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Format the results for the web interface
    results = {}
    for i in range(5):
        class_name = model.config.id2label[top5_indices[i].item()]
        # Gradio expects probabilities as decimals between 0 and 1
        confidence = float(top5_prob[i].item()) 
        results[class_name] = confidence
        
    return results

# Build the web interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    title="Apple MobileViT Image Classifier",
    description="Drag and drop a photo to see what the model thinks it is."
)

# Launch the local server
demo.launch()
