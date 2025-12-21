# RAG Assistant (Groq + ChromaDB)
A Safety-Aware Document-Grounded Chatbot using Groq and ChromaDB
![RAG Cover](images/rag_cover.png)

📌 Project Description

This project implements a Retrieval-Augmented Generation (RAG) assistant that combines vector-based document retrieval with large language model (LLM) inference to produce grounded, context-aware responses.

Unlike standalone LLMs that may hallucinate, this system retrieves relevant information from an external document corpus before generating answers, improving factual accuracy, transparency, and reliability.
The system also incorporates basic safety guardrails to support responsible usage.

🔎 Project Overview

The RAG assistant follows a modular pipeline:
1.Document ingestion and preprocessing
2.Dense vector embedding and persistent storage
3.Safety-aware query handling
4.Similarity-based retrieval
5.Context-conditioned response generation
This design makes the system suitable for academic, research, and domain-specific applications.

📂 Project Structure
project/
├── LICENSE
├── app.py                  # Main application entry point
├── vectordb.py             # Vector database (ChromaDB) wrapper
├── safety.py               # Safety and content guardrails
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── images/                 # Visual assets
│   ├── rag_cover.png
│   ├── rag_architecture.png
│   ├── methodology_flow.png
├── data/                   # Input .txt documents
└── rag_clients_db/         # Persistent ChromaDB storage

🏗️ System Architecture
![Architecture](images/rag_architecture.png)

🧠 Methodology
![Workflow](images/methodology_flow.png)

🔐 Safety & Ethical Considerations
To ensure responsible usage, the system includes:
 -Rule-based filtering for unsafe or restricted queries
 -Conservative fallback responses when information is unavailable
 -Prompt instructions to avoid hallucination
These safeguards provide a foundation for future enhancements such as neural moderation and audit logging.

🛠 Prerequisites
 -Python 3.10 or higher
 -Groq API key
 -Basic Python knowledge
 -Sufficient memory for embeddings

🚀 Step-by-Step Usage Guide
1️⃣ Clone the Repository
git clone <repository-url>
cd project
2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\Activate.ps1    # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Configure Environment Variables
GROQ_API_KEY=your_api_key_here
5️⃣ Add Documents

Place non-empty .txt files inside the data/ directory.

6️⃣ Run the Application
python app.py

📌 Applications
 - Domain-specific question answering
 - Academic research assistants
 - Enterprise knowledge bases
 - Documentation chatbots
 - Controlled AI assistants

⚠️ Limitations
 - Safety filtering is rule-based and limited
 - No explicit source citation in responses
 - Response quality depends on document coverage
 - No multi-hop reasoning across documents

🔮 Future Enhancements
 - Neural content moderation
 - Citation-aware responses
 - Multi-document reasoning
 - Web-based user interface
 - Quantitative evaluation benchmarks

📄 License & Usage

This project is intended for educational and research purposes.
Users are responsible for ensuring ethical, legal, and compliant usage in real-world deployments.
