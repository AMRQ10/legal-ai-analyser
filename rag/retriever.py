import chromadb
from chromadb.config import Settings
from typing import List
import os

class VectorRetriever:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initializes ChromaDB with persistent storage.
        Documents survive server restarts.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
    
    def get_or_create_collection(self, collection_name: str):
        """
        Gets existing collection or creates a new one.
        Each document gets its own collection.
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def store_chunks(
        self, 
        collection_name: str,
        chunks: List[dict],
        embeddings: List[List[float]]
    ) -> None:
        """
        Stores chunks and their embeddings in ChromaDB.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        collection = self.get_or_create_collection(collection_name)

        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks]
        )

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5

    ) -> List[str]:
        """
        Finds the n most relevant chunks for a given query embedding.
        Returns the chunk texts ordered by relevance.
        """
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            raise ValueError(
                f"Document '{collection_name}' not found."
                f"Please upload the document first."
            )
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count())
        )

        return results["documents"][0]

    def document_exists(self, collection_name: str) -> bool:
        """Checks if a document has already been proecessed and stored."""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def delete_document(self, collection_name: str) -> None:
        """Removes a document and all its vectors from the vector store."""
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            raise ValueError(f"Document '{collection_name}' not found.")
    
    def list_documents(self) -> List[str]:
        """Returns names of all stored documents."""
        collections = self.client.list_collections()
        return [col.name for col in collections]
