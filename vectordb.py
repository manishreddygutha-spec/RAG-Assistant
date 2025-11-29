import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "rag_clients_db")

os.makedirs(CHROMA_DIR, exist_ok=True)


class VectorDB:
    def __init__(self, collection_name, embedding_model_name):
        self.collection_name = collection_name

        # Chroma persistent DB path
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )

        # Collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"📚 VectorDB ready → {collection_name}")

    def chunk_text(self, text, chunk_size=1000):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        return splitter.split_text(text)

    def add_documents(self, documents):
        all_chunks = []
        metadata = []

        for i, doc in enumerate(documents):
            chunks = self.chunk_text(doc)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "doc_id": i,
                    "chunk_id": j,
                    "size": len(chunk)
                })

        print(f"🧩 Total chunks: {len(all_chunks)}")

        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(all_chunks)

        # Generate IDs
        start_id = self.collection.count()
        ids = [f"chunk_{start_id + i}" for i in range(len(all_chunks))]

        # Insert into Chroma
        self.collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=metadata
        )

        print("✅ Chunks added to vector DB.")

    def search(self, query, n_results=5):
        if not query:
            raise ValueError("Query cannot be empty")

        query_vector = self.embedding_model.embed_query(query)

        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )

        return {
            "documents": result["documents"][0],
            "distances": result["distances"][0],
            "metadatas": result["metadatas"][0]
        }
