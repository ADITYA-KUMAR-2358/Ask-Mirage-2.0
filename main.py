# Copyright (c) 2025 [ADYTA KUMAR]
# All Rights Reserved.

"""
This is an offline chatbot that processes PDF documents and generates responses
based on their content. It utilizes a local Large Language Model (LLM) for inference
and FAISS for efficient similarity search.

"""

from pypdf import PdfReader
from llama_cpp import Llama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class PDFProcessor:
    """
    Handles loading and extracting text from a PDF file.
    This class reads a PDF document from the specified folder and extracts its textual content.
    """
    def __init__(self, data_folder):
        # Initialize with the folder containing PDF files
        self.data_folder = data_folder
        self.pdf_file = self.select_pdf_file()
        self.text = self.read_pdf()
    
    def select_pdf_file(self):
        """
        Ask the user to pick a PDF file from the available options.
        """
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        if not pdf_files:
            raise FileNotFoundError("No PDF files found in the specified folder.")
        
        print("Available PDF files:")
        for i, file in enumerate(pdf_files):
            print(f"{i + 1}. {file}")
        
        choice = int(input("Enter the number of the PDF you want to use: ")) - 1
        return os.path.join(self.data_folder, pdf_files[choice])
    
    def read_pdf(self):
        """
        Extract text from the selected PDF file and return it as a single string.
        """
        print(f"Reading: {self.pdf_file}")
        reader = PdfReader(self.pdf_file)
        text_data = "\n".join(page.extract_text() or "" for page in reader.pages)
        print(f"Extraction complete: {len(text_data)} characters extracted.")
        return text_data

class TextProcessor:
    """
    Handles text chunking for embeddings.
    This class splits the extracted text into smaller chunks to improve retrieval efficiency.
    """
    def __init__(self, text):
        # Ask user for chunking parameters
        self.chunk_size = int(input("Enter chunk size (default: 500): ") or 500)
        self.chunk_overlap = int(input("Enter chunk overlap (default: 50): ") or 50)
        self.text = text
        self.chunks = self.split_text()
    
    def split_text(self):
        """
        Split the extracted text into smaller overlapping chunks for better retrieval.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_text(self.text)
        print(f"Total chunks created: {len(chunks)}")
        return chunks

class FAISSIndexer:
    """
    Creates and manages a FAISS index for efficient retrieval of text embeddings.
    This class generates numerical embeddings of the text chunks and stores them in a FAISS index.
    """
    def __init__(self, data_folder, chunks):
        # Initialize with folder path and text chunks
        self.data_folder = data_folder
        self.chunks = chunks
        self.embedding_model_name = "all-MiniLM-L6-v2"  # Set as default, no user input required
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.index = self.create_faiss_index()
    
    def create_faiss_index(self):
        """
        Generate vector embeddings for text chunks and create a FAISS index for similarity search.
        """
        print("Creating FAISS index...")
        chunk_embeddings = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings.astype(np.float32))
        faiss.write_index(index, os.path.join(self.data_folder, "pdf_faiss.index"))
        print("FAISS index saved.")
        return index
    
    def get_relevant_context(self, query, k=3):
        """
        Retrieve the most relevant text chunks based on the user's query.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
        D, I = self.index.search(query_embedding, k)
        return "\n".join([self.chunks[i] for i in I[0] if i < len(self.chunks)])

class PDFChatbot:
    """
    Offline chatbot that answers queries using context extracted from PDFs.
    This chatbot processes PDFs, creates an embedding-based search index, and responds to queries.
    """
    def __init__(self, data_folder, model_file):
        # Initialize with the data folder containing PDFs and the LLaMA model file
        self.data_folder = data_folder
        self.model_file = model_file
        self.llm = Llama(model_path=os.path.join(self.data_folder, self.model_file), n_ctx=4096)
        
        # Initialize components for PDF processing, text chunking, and indexing
        self.pdf_processor = PDFProcessor(self.data_folder)
        self.text_processor = TextProcessor(self.pdf_processor.text)
        self.faiss_indexer = FAISSIndexer(self.data_folder, self.text_processor.chunks)
    
    def chat(self, query):
        """
        Generate a response to a user query based on relevant text extracted from the PDF.
        """
        context = self.faiss_indexer.get_relevant_context(query)
        prompt = f"""
        You are an AI assistant. Based on the following text, answer the question concisely:
        
        Context: {context}
        
        Question: {query}
        
        Answer:"""
        response = self.llm(prompt, max_tokens=700)
        return response["choices"][0]["text"].strip()
    
    def start_chat(self):
        """
        Start an interactive chatbot session where users can ask questions.
        """
        print("Chatbot is ready! Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Chatbot: Goodbye!")
                break
            response = self.chat(user_input)
            print("Chatbot:", response)

if __name__ == "__main__":
    # Prompt user for the data folder containing PDFs and the LLaMA model file
    chatbot = PDFChatbot(
        data_folder=input("Enter the path to the data folder: "),
        model_file=input("Enter the LLaMA model filename: ")
    )
    chatbot.start_chat()
