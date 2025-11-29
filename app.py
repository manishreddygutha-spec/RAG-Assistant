import os
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tqdm import tqdm
from vectordb import VectorDB
import traceback
from langchain_community.document_loaders import TextLoader

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
api_key = os.getenv("GROQ_API_KEY")



def load_documents(documents_path) -> list:
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"Data folder not found: {documents_path}")

    results = []
    failed = []

    for filename in os.listdir(documents_path):
        if not filename.lower().endswith(".txt"):
            continue

        file_path = os.path.join(documents_path, filename)
        print(f"\n⏳ Trying to load: {file_path}")

        # First try the TextLoader (preferred)
        try:
            loader = TextLoader(file_path)  # if TextLoader supports encoding you can pass encoding='utf-8'
            loaded = loader.load()
            results.extend(loaded)
            print(f"✅ Loaded with TextLoader: {filename}")
            continue
        except Exception as e:
            print(f"⚠ TextLoader failed for {filename}: {e}")
            # continue to fallback readers

        # Fallback: try opening with utf-8, then latin-1, then binary decode with errors='replace'
        encodings_to_try = ["utf-8", "latin-1"]
        success = False
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read()
                # wrap into a simple object similar to loader output if needed by later code:
                class Doc:
                    def __init__(self, page_content):
                        self.page_content = page_content
                results.append(Doc(text))
                print(f"✅ Loaded with encoding {enc}: {filename}")
                success = True
                break
            except Exception as e:
                print(f"  - failed with encoding {enc}: {e}")

        if not success:
            try:
                # last resort: read bytes, decode with replacement to avoid errors
                with open(file_path, "rb") as f:
                    raw = f.read()
                text = raw.decode("utf-8", errors="replace")
                class Doc:
                    def __init__(self, page_content):
                        self.page_content = page_content
                results.append(Doc(text))
                print(f"✅ Loaded with binary fallback (utf-8 replace): {filename}")
                success = True
            except Exception as e:
                print(f"❌ All fallbacks failed for {filename}: {e}")
                traceback.print_exc()
                failed.append((filename, str(e)))

    print(f"\n📄 Total docs loaded: {len(results)}")
    if failed:
        print("\n⚠ Files that failed to load:")
        for fn, err in failed:
            print(f" - {fn} : {err}")

    # Convert to list of strings like your earlier code expected
    content = [d.page_content if hasattr(d, "page_content") else str(d) for d in results]
    return content

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=api_key
)


class RAGAssistant:
    def __init__(self):
        self.llm = llm
        self.vector_db = VectorDB(
            collection_name="rag_docs",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful RAG assistant.
Use only the given context to answer the question.

Context:
{context}

Question: {question}

Guidelines:
- If answer not found, respond: "I don't have this information"
- Do NOT hallucinate.
- Do NOT mention the context source.
""")

        self.chain = self.prompt_template | self.llm | StrOutputParser()
        print("\n✅ RAG system initialized.\n")

    def add_documents(self, documents):
        if not documents:
            print("❌ No documents provided.")
            return

        self.vector_db.add_documents(documents)
        print("📌 Documents added to vector database.")

    def search(self, query, n_results=5):
        return self.vector_db.search(query, n_results)

    def ask(self, question: str) -> str:
        results = self.search(question)

        if not results["documents"]:
            return "I don't have this information."

        context = "\n---\n".join(results["documents"][:2])

        return self.chain.invoke({
            "context": context,
            "question": question
        })


def main():
    try:
        print("🚀 Starting RAG Assistant...")

        assistant = RAGAssistant()

        print("📥 Loading documents...")
        docs = load_documents(DATA_DIR)

        assistant.add_documents(docs)

        print("\n🎉 RAG Assistant ready!\n")

        while True:
            question = input("Ask a question ('quit' to exit): ")

            if question.lower() == "quit":
                break

            print("\n⏳ Thinking...\n")
            print(assistant.ask(question))
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()
