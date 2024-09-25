from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import CSVLoader, TextLoader
from dotenv import load_dotenv
import shutil
from transformers import pipeline
import cv2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import numpy as np
import tempfile
import docx
from docx2txt import process

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load environment variables
load_dotenv()

hg_key = os.getenv("HUGGING_FACE_KEY")

# Initialize the language model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=600,
    top_k=25,
    temperature=0.3,
    repetition_penalty=1.01,
    huggingfacehub_api_token=hg_key,
)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

processed_files = set()

def load_document(file_path):
    """Load and read a document based on its file type."""
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        return load_img(file_path)
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    elif file_path.endswith('.csv'):
        return load_csv(file_path)
    elif file_path.endswith('.txt'):
        return load_txt(file_path)
    else:
        raise ValueError("Unsupported file type")

def load_pdf(file_path):
    """Load and extract text from a PDF file, including images."""
    try:
        images = convert_from_path(file_path)
        text = []
        for image in images:
            gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            ocr_text = pytesseract.image_to_string(thresh_image)
            text.append({'page_content': ocr_text})
        return text
    except Exception as e:
        print(f"Error loading PDF file: {e}")
        return []


def preprocess_image(image_path):
    """Preprocess the image to enhance OCR accuracy."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Resize image to make text larger
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Step 2: Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Step 3: Apply adaptive thresholding to binarize the image
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
    
    # Step 4: Increase the contrast
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return img


def load_docx(file_path):
    """Load and extract text and images from a DOCX file."""
    try:
        # Extract text from DOCX using docx2txt
        text = process(file_path)
        extracted_text = [{'page_content': text}]
        
        # Load the DOCX file using python-docx
        doc = docx.Document(file_path)
        
        # Create a temporary directory to save images
        temp_dir = tempfile.mkdtemp()
        
        # Iterate over the document and extract images
        for i, rel in enumerate(doc.part.rels):
            if "image" in doc.part.rels[rel].target_ref:
                image = doc.part.rels[rel].target_part
                image_bytes = image.blob
                image_path = os.path.join(temp_dir, f"image{i}.jpeg")
                
                # Save the image temporarily
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Preprocess the image before OCR
                preprocessed_img = preprocess_image(image_path)
                
                # Perform OCR on the preprocessed image
                ocr_text = pytesseract.image_to_string(preprocessed_img)
                extracted_text.append({'page_content': ocr_text})
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return extracted_text
    except Exception as e:
        print(f"Error loading DOCX file: {e}")
        return []

def load_csv(file_path):
    """Load and extract text from a CSV file."""
    try:
        loader = CSVLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

def load_txt(file_path):
    """Load and extract text from a TXT file."""
    try:
        loader = TextLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading TXT file: {e}")
        return []

def load_img(file_path):
    """Load and extract text from an image file."""
    try:
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh_image)
        return [{'page_content': text}]
    except Exception as e:
        print(f"Error loading image file: {e}")
        return []

def compare_documents(message, documents):
    """Compare user message with the content of uploaded documents."""
    full_text = "\n".join([doc['page_content'] if isinstance(doc, dict) else doc.page_content for doc in documents])

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    texts = text_splitter.split_text(full_text)

    # Create embeddings and a FAISS database
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever()

    # Define the prompt template
    template = """Answer the question based only on the following context: {context}. 
    Reject questions unrelated to the context: {context}.
    Note: Be accurate and concise. Don't create your own question. Just answer the user's chat.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = llm

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )    

    response = chain.invoke(message)

    pattern = r"Question:.*"
    cleaned_response = re.sub(pattern, '', response).strip()

    return cleaned_response

#--------------------------------------------------------classify content
def classify_content(content):
    """Classify the content of the document."""
    truncated_content = content[:512]

    if not truncated_content.strip():
        raise ValueError("The content is empty after truncation.")
    
    labels = ["Technology", "Business", "Science", "Health", "Entertainment", "Sports"]
    
    classifier = pipeline("zero-shot-classification",
                          model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", 
                          model_max_length=512,
                          truncation=True)

    result = classifier(truncated_content, labels)
    
    if 'labels' not in result or not result['labels']:
        raise ValueError("No labels found in classification result.")
    
    return result['labels'][0]

def move_file(file_path, category):
    """Move the file to the appropriate category directory."""
    category_dir = os.path.join(app.config['UPLOAD_FOLDER'], category)
    
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
        
    shutil.move(file_path, os.path.join(category_dir, os.path.basename(file_path)))

def process_directory(file_path):
    """Process the file and move it based on its content classification."""
    documents = load_document(file_path)
    full_text = "\n".join([doc['page_content'] if isinstance(doc, dict) else doc.page_content for doc in documents])

    if full_text:
        category = classify_content(full_text)
        move_file(file_path, category)
        print(f"Moved {file_path} to {category} category")
    else:
        print(f"Skipped {file_path}: Unsupported format or empty content")

#----------------------------------------------------------------------
#DISPLAY FOLDER
# Function to recursively get folder structure
def get_folder_structure(path):
    folder_structure = {}
    for root, dirs, files in os.walk(path):
        relative_path = os.path.relpath(root, path)
        if relative_path == '.':
            # Skip adding the root folder itself
            continue
        folder_name = relative_path
        folder_structure[folder_name] = {
            "dirs": dirs,
            "files": files
        }
    return folder_structure


@app.route('/', methods=['GET'])
def index():
    folder_structure = get_folder_structure(UPLOAD_FOLDER)
    return render_template('index2.html') #----remove after the creation of html
    # return render_template('index2.html', folder_structure)

@app.route('/upload', methods=['POST'])
def upload():
    user_message = request.form.get('chat', '') + ' in the document'
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'response': 'No file selected.'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check if the file has been processed
        if file_path not in processed_files:
            # Load and extract te  xt from the uploaded document
            documents = load_document(file_path)
            
            # Get the bot's response
            bot_message = compare_documents(user_message, documents)

            # Process the file and move it to the appropriate directory
            process_directory(file_path)
            
            # Mark the file as processed
            processed_files.add(file_path)

            # Extract the answer from the bot's response
            pattern = r"Answer:\s*(.*)"
            match = re.search(pattern, bot_message, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                answer = "Sorry, your input is not related to the document."
        else:
            # File has been processed, use the existing document
            documents = load_document(file_path)
            bot_message = compare_documents(user_message, documents)
            pattern = r"Answer:\s*(.*)"
            match = re.search(pattern, bot_message, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                answer = "Sorry, your input is not related to the document."

        return jsonify({'response': answer})

    return jsonify({'response': 'No file uploaded.'}), 400

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/signin', methods=['GET'])
def signin():
    return render_template('signin.html')

@app.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('abot.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)