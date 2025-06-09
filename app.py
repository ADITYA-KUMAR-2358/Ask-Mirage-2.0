from flask import Flask, request, jsonify, render_template
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = "uploads/"
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pdf_faiss.index')

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LLM Model (Optimized for 16GB RAM & 4GB VRAM)
print("Loading LLM Model...")
print(f"Checking for model at: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=3,
        n_batch=32,
        n_ctx=1024,
        verbose=False
    )
    print("LLM model loaded successfully.")
else:
    llm = None
    print(f"Model not found at {MODEL_PATH}")

# Load Sentence Transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Global variable to store chunks
global_chunks = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    return jsonify({"success": True, "filename": file.filename})

@app.route("/train", methods=["POST"])
def train_model():
    global global_chunks  # Use the global variable to store chunks
    data = request.get_json()
    filename = data.get("filename")
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return jsonify({"error": f"File not found: {filepath}"})
    
    reader = PdfReader(filepath)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    global_chunks = text_splitter.split_text(text)  # Store chunks globally

    if not global_chunks:
        return jsonify({"error": "No text extracted from PDF"})

    # Generate embeddings
    embeddings = embedding_model.encode(global_chunks, convert_to_numpy=True).astype(np.float32)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_PATH)

    return jsonify({"success": True, "chunks": len(global_chunks)})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("query")

    if not os.path.exists(INDEX_PATH):
        return jsonify({"error": "Train the model first"})

    index = faiss.read_index(INDEX_PATH)
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = index.search(query_embedding, 3)  # Retrieve top 3 matches

    # Validate FAISS results
    if I is None or len(I[0]) == 0 or all(i == -1 for i in I[0]):
        return jsonify({"error": "No relevant data found in the index"})

    # Retrieve the original chunks based on the index
    retrieved_chunks = []
    for i in I[0]:
        if i != -1:
            retrieved_chunks.append(global_chunks[i])  # Use the global chunks

    retrieved_text = "\n".join(retrieved_chunks)

    # Format prompt
    full_prompt = f"""
    You are an AI assistant. Answer the question based on the provided context.

    ### Context:
    {retrieved_text}

    ### Question:
    {query}

    ### Answer:
    """

    # Generate response
    if llm:
        response = llm(full_prompt, max_tokens=200)
        response_text = response["choices"][0]["text"].strip()
    else:
        response_text = "LLM Model not loaded."

    return jsonify({"answer": response_text})

if __name__ == "__main__":
    app.run(debug=True)