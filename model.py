import pytesseract
import cv2
import numpy as np
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, AutoProcessor
import torch

# Load your trained model and processor
model_name = r"/home/aswin/LayoutLM/model"
processor = AutoProcessor.from_pretrained(model_name)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

def normalize_bbox(bbox, width, height):
    """Normalize bounding boxes to a 0-1000 scale."""
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
        max(0, min(1000, int(1000 * x2 / width))),
        max(0, min(1000, int(1000 * y2 / height)))
    ]

def extract_text_and_boxes(image):
    """Extract text and bounding boxes using Tesseract OCR."""
    image = Image.fromarray(image).convert("RGB")
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words, boxes = [], []
    image_width, image_height = image.size
    
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        if word:
            x1 = ocr_data['left'][i]
            y1 = ocr_data['top'][i]
            x2 = x1 + ocr_data['width'][i]
            y2 = y1 + ocr_data['height'][i]
            
            normalized_box = normalize_bbox([x1, y1, x2, y2], image_width, image_height)
            words.append(word)
            boxes.append(normalized_box)
    
    return words, boxes

def predict_tags(image, words, boxes):
    """Predict tags using the trained LayoutLMv3 model."""
    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    label_map = model.config.id2label
    predicted_tags = [label_map[pred] for pred in predictions]
    
    return predicted_tags[:len(words)]  # Ensure correct alignment

def draw_bounding_boxes(image, words, tags, boxes):
    """Draw bounding boxes with predicted tags on the image."""
    image_with_boxes = image.copy()
    
    for (word, tag, (x1, y1, x2, y2)) in zip(words, tags, boxes):
        # Convert normalized bbox to image size
        x1, y1, x2, y2 = int(x1 * image.shape[1] / 1000), int(y1 * image.shape[0] / 1000), int(x2 * image.shape[1] / 1000), int(y2 * image.shape[0] / 1000)
        
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{word} ({tag})"
        cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image_with_boxes

def resize_image(image, target_size=(1024, 1024)):
    """Resize image to a standard size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def extract_text_and_tags(image):
    """Extract text, bounding boxes, and predict tags."""
    words, boxes = extract_text_and_boxes(image)
    tags = predict_tags(image, words, boxes)
    return words, tags, boxes
