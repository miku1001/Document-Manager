from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
from dotenv import load_dotenv
import os
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
import fitz  # PyMuPDF
from docx import Document

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Initialize the model with the API key
model = ChatMistralAI(model="mistral-large-latest", api_key=mistral_api_key)

app = Flask(__name__)

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)  # You should use a more secure secret key in production
Session(app)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_document(file_path):
    """Load and read a document based on its file type."""
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    elif file_path.endswith('.csv'):
        return load_csv(file_path)
    else:
        raise ValueError("Unsupported file type")

def load_pdf(file_path):
    """Load and read a PDF file."""
    pdf_document = fitz.open(file_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

def load_docx(file_path):
    """Load and read a DOCX file."""
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def load_csv(file_path):
    """Load and read a CSV file."""
    text = ""
    with open(file_path, 'r') as file:
        for line in file:
            text += line
    return text

def interact_with_document(document_content, chat, is_file_uploaded):
    # Define behavior for the AI assistant
    behavior = """You are an AI assistant that will assist the user with documents. If the user does not upload a file, respond to greetings and general conversation appropriately. If the user asks a question related to document management, provide assistance based on the document content. If the user asks something unrelated to document management, inform them that the system only handles document management. Be accurate, concise, and casual in your responses."""
    
    # Initialize conversation history from session
    user_history = session.get('history', [])
    
    # Construct the prompt based on whether a file is uploaded or not
    if not is_file_uploaded:
        if any(greeting in chat.lower() for greeting in ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']):
            prompt = f"{behavior} {chat} Hello! I'm LiamBot. I can help you manage your files. Would you like to upload a file to get started?"
        else:
            prompt = f"{behavior} {chat} The system only handles document management. Please upload a document to get assistance."
    else:
        prompt = behavior + f"{chat} Document: {document_content}"

    # Append the new message to history
    user_history.append(HumanMessage(content=prompt))

    # Invoke the model with the prompt
    response = model.invoke(user_history)
    
    # Extract and return the AI's response
    cleaned_response = response.content

    # Append the AI's response to history
    user_history.append(AIMessage(content=cleaned_response))
    
    # Save the updated history back to the session
    session['history'] = user_history

    return cleaned_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is provided
    if 'file' not in request.files:
        chat = request.form.get('chat', '')
        response = interact_with_document("", chat, False)
        return jsonify({"response": response})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    document_content = load_document(file_path)
    chat = request.form.get('chat', '')

    response = interact_with_document(document_content, chat, True)

    return jsonify({"response": response})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)