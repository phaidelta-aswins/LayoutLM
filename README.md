# OCR & Text Tagging with LayoutLMv3

This project implements an OCR system using `Tesseract` and `LayoutLMv3` for text extraction and tagging from images. The extracted text is processed and displayed in a user-friendly interface using `Gradio`, with an option to store the results in an SQLite database.

## Features
- Extract text from images using `Tesseract OCR`
- Predict entity tags using `LayoutLMv3`
- Display results with bounding boxes on images
- Interactive `Gradio` UI for easy processing
- Store extracted text and metadata in an SQLite database
- Preprocessing support for noisy images

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip
- Tesseract OCR (`pytesseract`)
- SQLite (comes pre-installed with Python)

### Install Dependencies

```bash
pip install torch==2.6.0 transformers==4.36.2 pytesseract==0.3.10 opencv-python==4.9.0.80 numpy==1.26.3 Pillow==10.2.0 gradio==5.16.0 pip==25.0.1
```

### Setup Tesseract OCR
Ensure Tesseract OCR is installed and added to your system's PATH. For Ubuntu, run:

```bash
sudo apt update && sudo apt install tesseract-ocr
```

For Windows, download from: [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

---

## Project Structure

```
Myproject/
│── LAYOUTLM/               # Main project folder
│   │── model/              # Contains model files
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── tokenizer.json
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── training_args.bin
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │── model.py            # Model loading & text extraction
│   │── ui.py               # Gradio UI for image processing
│   │── database.py         # SQLite database management
│── .gitignore              # Git ignore file
│── README.md               # Project documentation
```

---

## Usage

### Running the Application
Launch the `Gradio` UI by executing:

```bash
python ui.py
```

This will start a local web server where you can upload images for processing.

### Processing an Image
1. Upload an image via the UI.
2. Click `Process Image` to extract text and tags.
3. If the image is noisy, click `Preprocess Image` to clean it before reprocessing.
4. Review the extracted words and tags in a structured table.
5. Save results to the database by clicking `Save to Database`.

---

## Explanation of Key Files

### **1. `model.py` (Text Extraction & Tagging)**
- Loads the trained `LayoutLMv3` model and processor.
- Extracts text and bounding boxes using `pytesseract`.
- Normalizes bounding boxes for processing.
- Uses the `LayoutLMv3` model to predict entity tags.

### **2. `ui.py` (Gradio Web Interface)**
- Provides an interactive UI for uploading and processing images.
- Implements noise detection and preprocessing for better OCR accuracy.
- Displays extracted text and tags with bounding boxes.
- Saves extracted data into a database upon user confirmation.

### **3. `database.py` (SQLite Database Management)**
- Creates and initializes an SQLite database (`ocr_data1.db`).
- Stores extracted words, bounding boxes, and predicted tags.
- Supports inserting extracted data into the database.

---

## Example Output
After processing an image, the UI displays:

- The original image with bounding boxes around extracted text.
- A structured table of words and their predicted tags.
- A status message indicating the image quality.

---

## Future Enhancements
- Add support for more image preprocessing techniques.
- Improve noise detection and adaptive thresholding.
- Extend database functionality to support searching and filtering.

---

## License
This project is open-source under the MIT License.

---

## Author
[Your Name]

For questions, feel free to reach out!

