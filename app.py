import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from safety import validate_query

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in environment variables.")


def load_documents(documents_path: str) -> list:
    """
    Loads non-empty .txt documents from the given directory.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        RuntimeError: If no valid documents are found.
    """
    if not os.path.exists(documents_path):
        raise FileNotFoundError(
            f"Data directory not found: {documents_path}"
        )

    documents = []

    for filename in os.listdir(documents_path):
        if not filename.lower().endswith(".txt"):
            continue

        file_path = os.path.join(documents_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if text.strip():
                documents.append(text)
            else:
                print(f"⚠ Skipping empty file: {filename}")

        except Exception as e:
            print(f"⚠ Failed to read {filename}: {e}")

    print(f"📄 Loaded {len(documents)} documents.")

    if not documents:
        raise RuntimeError(
            "No valid .txt documents found in data/. "
            "Please add at least one non-empty .txt file."
        )

    return documents


# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=GROQ_API_KEY
)


class RAGAssistant:
    """
    Retrieval-Augmented Generation (RAG) assistant that integrates
    vector-based retrieval with LLM-based response generation.
    """

    def __init__(self):
        self.vector_db = VectorDB(
            collection_name="rag_docs",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are a helpful RAG assistant.
Use only the provided context to answer the question.

Context:
{context}

Question: {question}

Guidelines:
- Do not hallucinate.
- If the answer is not present in the context, reply with:
  "I don't have this information."
"""
        )

        self.chain = self.prompt_template | llm | StrOutputParser()
        print("✅ RAG Assistant initialized.")

    def ask(self, query: str) -> str:
        # Stage 1: Safety validation
        validate_query(query)

        # Stage 2: Similarity-based retrieval
        results = self.vector_db.search(query)

        if not results["documents"]:
            return "I don't have this information."

        # Stage 3: Context aggregation
        context = "\n---\n".join(results["documents"][:2])

        # Stage 4: LLM-based response generation
        return self.chain.invoke({
            "context": context,
            "question": query
        })


def main():
    print("🚀 Starting RAG Assistant...")

    # Load documents
    documents = load_documents(DATA_DIR)

    # Initialize assistant
    assistant = RAGAssistant()

    # Index documents
    assistant.vector_db.add_documents(documents)

    print("\n🎉 RAG Assistant is ready!\n")

    while True:
        query = input("Ask a question (type 'quit' to exit): ").strip()

        if query.lower() == "quit":
            print("👋 Exiting RAG Assistant.")
            break

        try:
            print("\n⏳ Thinking...\n")
            answer = assistant.ask(query)
            print(answer)
            print()
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
