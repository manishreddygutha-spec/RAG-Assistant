# RAG Assistant (Groq + ChromaDB)

This repository contains a small Retrieval-Augmented Generation (RAG) chatbot built with Groq (via `langchain-groq`) and a local ChromaDB vector store.  
The codebase includes two main modules:

- `app.py` — main application that loads documents, initializes the LLM (ChatGroq), and runs the ask/search loop.  
- `vectordb.py` — vector database wrapper that handles chunking, embeddings (via `sentence-transformers`/HuggingFaceEmbeddings) and storage in a ChromaDB collection.

> **Purpose:** Create a RAG chatbot using Groq for LLM responses and ChromaDB for retrieval of document context.

---

## Project structure

project/
├─ app.py
├─ vectordb.py
├─ requirements.txt
├─ .gitignore
├─ README.md
├─ data/ # Put your .txt files here (NOT committed)
└─ rag_clients_db/ # ChromaDB persistent files (auto-created, NOT committed)

## Prerequisites
- Python 3.10+ recommended
- A Groq API key (set as `GROQ_API_KEY` in a `.env` file or environment variable)
- Enough RAM for embeddings/LLM operations (depending on model sizes)

---

## Setup

1. Clone or copy this repository to your machine.

2. Create & activate a virtual environment:
```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

3. Install dependencies:
pip install -r requirements.txt

4. Create a .env file at project root with your Groq API key:
GROQ_API_KEY=your_real_groq_api_key_here

5. Add documents:
Create a data/ folder at project root and drop plain .txt files to be used as knowledge sources.4

6. Running the app
python app.py

Follow prompts in the terminal to ask questions. The app will:
    1. Load .txt files from data/.
    2. Split text into chunks, embed them, and store embeddings in rag_clients_db/.
    3. Use ChatGroq to generate responses informed by the top retrieved chunks.

