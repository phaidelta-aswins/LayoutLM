import gradio as gr
import numpy as np
import cv2
from PIL import Image
from database import insert_data
from model import extract_text_and_tags, draw_bounding_boxes

# Image Preprocessing Function
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise removal using median blur
    gray = cv2.medianBlur(gray, 3)

    # Adaptive thresholding to binarize the image
    processed_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Convert back to RGB
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    
    return processed_image

# Function to detect noise in the image
def detect_noise(image):
    # Calculate the variance of the Laplacian (a measure of noise)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()
    return noise_level

# Process image function (Check if image is good or needs preprocessing)
def process_image(image):
    # Ensure the image is in RGB format
    image = Image.fromarray(image).convert('RGB')
    image = np.array(image)

    # Check if the image is good based on brightness and noise level
    is_good_image = np.mean(image) > 150  # This is a simple heuristic, you can improve it
    noise_level = detect_noise(image)
    high_noise = noise_level > 100  # Adjust this threshold based on your requirements

    if is_good_image and not high_noise:
        words, tags, boxes = extract_text_and_tags(image)
        # Convert to table format for UI
        table_data = list(zip(words, tags))
        boxed_image = draw_bounding_boxes(image, words, tags, boxes)
        return boxed_image, table_data, words, tags, boxes, "✅ Image is good to go!"
    
    else:
        if high_noise:
            return None, None, None, None, None, "⚠️ High noise detected. Preprocessing needed."
        else:
            return None, None, None, None, None, "⚠️ Image needs preprocessing due to low brightness or noise."

# Function to preprocess and re-process the image
def preprocess_and_process_image(image):
    preprocessed_image = preprocess_image(image)
    words, tags, boxes = extract_text_and_tags(preprocessed_image)
    table_data = list(zip(words, tags))
    boxed_image = draw_bounding_boxes(preprocessed_image, words, tags, boxes)
    return boxed_image, table_data, words, tags, boxes, "✅ Image preprocessed and good to go!"

# Function to save extracted data to the database when flag is pressed
def save_to_db(image, words, tags, boxes):
    insert_data("uploaded_image", words, tags, boxes)
    return "✅ Data saved successfully!"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 📝 OCR & Tagging with LayoutLMv3")
    
    # Image Upload & Processing
    image_input = gr.Image(label="Upload Image", type="numpy")
    image_output = gr.Image(label="Processed Image")
    data_output = gr.Dataframe(headers=["Word", "Tag"], label="Extracted Data")
    status_output = gr.Textbox(label="Image Status")
    
    words_state = gr.State()
    tags_state = gr.State()
    boxes_state = gr.State()
    
    process_btn = gr.Button("🔍 Process Image")
    preprocess_btn = gr.Button("🔧 Preprocess Image", visible=False)  # Initially hidden
    flag_btn = gr.Button("🚩 Save to Database")

    # Image processing logic
    process_btn.click(process_image, inputs=[image_input], 
                      outputs=[image_output, data_output, words_state, tags_state, boxes_state, status_output])
    
    preprocess_btn.click(preprocess_and_process_image, inputs=[image_input], 
                         outputs=[image_output, data_output, words_state, tags_state, boxes_state, status_output])

    # Save data to the database
    flag_btn.click(save_to_db, inputs=[image_input, words_state, tags_state, boxes_state], 
                   outputs=gr.Textbox(label="Database Status"))

    # Show preprocess button if high noise is detected
    def show_preprocess_button(status):
        return gr.update(visible="⚠️ High noise detected" in status)
    
    status_output.change(show_preprocess_button, inputs=status_output, outputs=preprocess_btn)

app.launch(server_name="0.0.0.0", server_port=7860)
