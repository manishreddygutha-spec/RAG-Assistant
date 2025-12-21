import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "rag_clients_db")
os.makedirs(CHROMA_DIR, exist_ok=True)


class VectorDB:
    """
    Vector database layer for the Retrieval-Augmented Generation (RAG) system.

    Responsibilities:
    - Text chunking with overlap
    - Embedding generation using sentence-transformer models
    - Persistent vector storage using ChromaDB
    - Similarity-based retrieval
    """

    def __init__(self, collection_name: str, embedding_model_name: str):
        self.collection_name = collection_name

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"📚 VectorDB initialized → {collection_name}")

    def add_documents(self, documents: list):
        """
        Adds documents to the vector database after chunking and embedding.

        Raises:
            ValueError: If documents, chunks, or embeddings are empty.
        """
        if not documents:
            raise ValueError("No documents provided for indexing.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = []
        metadatas = []

        for doc_id, doc in enumerate(documents):
            if not isinstance(doc, str) or not doc.strip():
                continue

            split_chunks = splitter.split_text(doc)

            for chunk_id, chunk in enumerate(split_chunks):
                if not chunk.strip():
                    continue

                chunks.append(chunk)
                metadatas.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "length": len(chunk)
                })

        if not chunks:
            raise ValueError(
                "No valid text chunks were generated. "
                "Ensure documents contain readable text."
            )

        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(chunks)

        if not embeddings:
            raise ValueError("Embedding generation returned empty output.")

        # Generate unique IDs
        start_index = self.collection.count()
        ids = [f"chunk_{start_index + i}" for i in range(len(chunks))]

        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✅ Indexed {len(chunks)} chunks into vector database.")

    def search(self, query: str, n_results: int = 5) -> dict:
        """
        Performs similarity-based retrieval for a given query.

        Returns:
            dict: Retrieved documents and metadata.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.embedding_model.embed_query(query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return {
            "documents": result["documents"][0],
            "metadatas": result["metadatas"][0],
            "distances": result["distances"][0]
        }
