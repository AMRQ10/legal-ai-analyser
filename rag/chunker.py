from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        chunk_size: maximum characters per chunk
        chunk_overlap: characters shared between consecutive chunks

        Overlap is critical for legal documents - a clause often starts 
        near the end of one chunk and continues into the next. Overlap 
        ensures no clause gets cut off and lost between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, text: str, document_id: str) -> List[dict]:
        """
        Splits document text into chunks and returns them as a list
        of dicts with the chunks text and metadata.
        """
        if not text or not text.strip():
            raise ValueError("Document text cannot be empty")

        chunks = self.splitter.split_text(text)

        return [
            {
                "id": f"{document_id}_chunk_{i}",
                "text": chunk,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            } for i, chunk in enumerate(chunks)
        ]