# my-ai-mentor
This project is a free, web-based, AI-powered tool designed to act as a personalized and low-stress mentor for individuals with cognitive disabilities resulting from conditions such as stroke, PTSD, or ADHD. The tool focuses on helping users relearn basic functions, improve memory recall, and enhance focus in a supportive environment
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from huggingface_hub import hf_hub_download

# --- Data Persistence: Download the knowledge file ---
HF_DATASET_REPO = "your-username/your-knowledge-dataset" # Replace this with your own dataset repository
KNOWLEDGE_FILE = "knowledge_base.txt"
DOWNLOAD_DIR = "./data"

def download_knowledge_base():
    """Downloads the knowledge base file from the Hugging Face Hub."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=KNOWLEDGE_FILE,
        local_dir=DOWNLOAD_DIR
    )
    return file_path

# --- RAG Setup: Create the search engine ---
knowledge_file_path = download_knowledge_base()
with open(knowledge_file_path, "r", encoding="utf-8") as f:
    knowledge_base = f.read()
    
# Split the text into smaller, easier-to-search chunks
text_chunks = knowledge_base.split("\n\n")

# Load a free embedding model (converts text to numbers)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(text_chunks)

# Create a FAISS index to make searching very fast
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# --- AI Mentor Logic: Combine RAG and the LLM ---
# Load a small, fast LLM for the free tier
generator = pipeline("text-generation", model="gpt2") # Use gpt2 for a free example, but a more suitable model might be better.

def retrieve_and_generate(user_input):
    """Retrieves relevant info and generates a response."""
    # 1. RETRIEVE: Find relevant information from the custom knowledge base
    query_embedding = embedding_model.encode([user_input])
    distances, indices = index.search(np.array(query_embedding), k=1) # Get the top 1 most relevant chunk
    retrieved_chunk = text_chunks[indices[0][0]]
    
    # 2. AUGMENT: Add the retrieved information to the AI's instructions
    prompt = (
        f"You are a compassionate mentor for someone with memory issues. "
        f"Use the following information to guide the user, but do not directly quote it:\n"
        f"'{retrieved_chunk}'\n\n"
        f"User: {user_input}\n"
        f"Mentor:"
    )
    
    # 3. GENERATE: Let the LLM create a response based on the instructions
    response = generator(prompt, max_new_tokens=100, num_return_sequences=1)[0]['generated_text']
    
    # Clean up the output to remove the initial prompt
    return response.split("Mentor:")[1].strip()

# --- Gradio Interface: Present the mentor to the user ---
demo = gr.ChatInterface(
    fn=retrieve_and_generate,
    title="My Personalized AI Mentor"
)

demo.launch()
