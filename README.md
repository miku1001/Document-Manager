# LiamBot

This application allows users to upload documents (PDF, images, DOCX, CSV, TXT), extracts text content using OCR and other text-processing techniques, and compares a user’s input message to the contents of the uploaded document using embeddings and a FAISS-based search. The app also classifies the content and organizes the uploaded files into categories.

## Features

- **File Upload**: Supports various file formats like `.pdf`, `.png`, `.jpg`, `.jpeg`, `.docx`, `.csv`, and `.txt`.
- **OCR Integration**: Extracts text from images and PDF documents with `pytesseract`.
- **Text Comparison**: Compares user messages with the contents of the uploaded document using embeddings and FAISS.
- **Content Classification**: Classifies the document content into categories like Technology, Business, Science, etc.
- **Organized File Management**: Moves the processed files into category-specific directories.

## Installation

1. **Python 3.8+** installed.
2. Install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
3. Install required Python packages (listed below).

4. Clone repository
```bash
   git clone https://github.com/miku1001/Document-Manager   
   cd https://github.com/miku1001/Document-Manager
```

5. Install modules/ dependencies
```bash
pip install -r requirements.txt
```

## Usage
Rub the flask app or test5.py
```python
python test5.py
```

## File Pathing

Instruction is given in the pdf file