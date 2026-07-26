from typing import List, Optional
from rag.chunker import DocumentChunker
from rag.embedder import DocumentEmbedder
from rag.retriever import VectorRetriever
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os


load_dotenv()

class ClauseAnalysisOutput(BaseModel):
    plain_english: str = Field(description="Plain English explanation of the clause")
    obligations: List[str] = Field(description="List of obligations created by this clause")
    risks: List[str] = Field(description="List of potential risks or issues")
    risk_rating: str = Field(description="Risk level: LOW, MEDIUM, or HIGH")
    risk_score: int = Field(description="Numerical risk score from 1 to 10")
    recommendations: List[str] = Field(description="Recommended notifications")
    clause_type: str = Field(description="Type of clause e.g. Non-Compete, Confidentiality")

class DocumentSummaryOutput(BaseModel):
    document_type: str = Field(description="Type of document e.g. NDA, Employment Contact")
    parties: List[dict] = Field(description="List of parties with name and role")
    purpose: str = Field(description="One sentence description of document purpose")
    key_terms: List[str] = Field(description="List of key terms and conditions")
    concerns: List[str] = Field(description="List of concerns or problematic clauses")
    missing_protections: List[str] = Field(description="List of missing standard protections")
    overall_risk_rating: str = Field(description="Overall risk level: LOW, MEDIUM or HIGH")
    overall_risk_score: int = Field(description="Overall risk score from 1 to 10")
    summary: str = Field(description="2-3 sentence plain English summary")

class ClauseItem(BaseModel):
    clause_type: str = Field(description="Type of clause")
    verbatim_text: str = Field(description="Exact text from document")
    plain_english: str = Field(description="Simple explanation")
    risk_rating: str = Field(description="Risk level: LOW, MEDIUM or HIGH")
    risk_score: int = Field(description="Risk score from 1 to 10")
    concerns: List[str] = Field(description="List of concerns")

class ExtractClausesOutput(BaseModel):
    total_clauses_found: int = Field(description="Total number of clauses found")
    high_risk_count: int = Field(description="Number of high risk clauses")
    medium_risk_count: int = Field(description="Number of medium risk clauses" )
    low_risk_count: int = Field(description="Number of low risk clauses")
    clauses: List[ClauseItem] = Field(description="List of clauses ordered by risk score descending")

class QAOuput(BaseModel):
    answer: str = Field(description="Direct answer to the question")
    confidence: str = Field(description="Confidence level: HIGH, MEDIUM, or LOW")
    relevant_excerpt: str = Field(description="Most relevant quote from the document")
    legal_implications: List[str] = Field(description="List of legal implications")
    follow_up_questions: List[str] = Field(description="Suggest follow app questions")


class LegalAnalyser:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )

        self.system_prompt= """You are an expert legal analyst with deep 
knowledge of contract law, corporate law, and regulatory compliance.
When analysing legal text you:
- Identify key obligations and rights for each party
- Flag potentially problematic or unusual clauses
- Highlight missing standard protections
- Explain complex legal language in plain English
- Rate risk level as LOW, MEDIUM, or HIGH
- Assign a numerical risk score from 1-10"""

        self.chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        self.retriever = VectorRetriever()
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = DocumentEmbedder()
        return self._embedder
    
    def _get_chain(self, output_schema):
        """Creates a structured output chain for a given schema."""
        structured_llm = self.llm.with_structured_ouput(output_schema)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        return prompt | structured_llm

    def analyse_clause(self, clause_text: str) -> dict:
        if not clause_text or not clause_text.strip():
            raise ValueError("Clause text cannot be empty")

        clause_text = clause_text.replace('\r', ' ').replace('\n', ' ').strip()

        chain = self._get_chain(ClauseAnalysisOutput)

        try:
            result = chain.invoke({
                "input": f"Analyse this contract clause:\n\n{clause_text}"
            })
            return result.model_dump()
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")
        
    def summarise_document(self, document_text: str) -> dict:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")
        
        document_text = document_text.replace('\r', ' ').replace('\n', ' ').strip()

        chain = self._get_chain(DocumentSummaryOutput)

        try:
            result = chain.invoke({
                "input": f"Analyse this legal document:\n\n{document_text}"
            })
            return result.model_dump()
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")
    
    def extract_key_clauses(self, document_text: str) -> dict:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")
        
        document_text = document_text.replace('\r', ' ').replace('\n', ' ').strip()

        chain = self._get_chain(ExtractClausesOutput)

        try:
            result = chain.invoke({
                "input": f"""Extract and categorise all key clauses from this 
legal document. Order clauses by risk_score descending:\n\n{document_text}"""
            })
            return result.model_dump()
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")

    def index_document(self, document_text: str, document_id: str) -> dict:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
        
        collection_name = document_id.lower().replace(" ", "_").replace(".", "_")

        if self.retriever.document_exists(collection_name):
            return {
                "message": f"Document '{document_id}' already indexed",
                "document_id": document_id,
                "status": "already_exists"
            }

        chunks = self.chunker.chunk_document(document_text, collection_name)
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        self.retriever.store_chunks(collection_name, chunks, embeddings)

        return {
            "message": f"Document indexed successfully",
            "document_id": document_id,
            "chunks_created": len(chunks),
            "status": "indexed"
        }

    def answer_question(self, document_id: str, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        collection_name = document_id.lower().replace(" ", "_").replace(".", "_")

        query_embedding = self.embedder.embed_query(question)
        relevant_chunks = self.retriever.search(
            collection_name, 
            query_embedding, 
            n_results=5
        )

        context = "\n\n---\n\n".join(relevant_chunks)

        try:
            result = chain.invoke({
                "input": f"""Answer this question using ONLY the provided 
contract excerpts. If the answer cannot be found, set confidence to LOW.

QUESTION: {question}

CONTRACT EXCERPTS:
{context}"""
            })
            output = result.model_dump()
            output["document_id"] = document_id
            output["question"] = question
            return output
        except Exception as e:
            raise RuntimeError(f"Q&A failed: {e}")
        
    def list_indexed_documents(self) -> List[str]:
        return self.retriever.list_documents()

    def delete_indexed_document(self, document_id: str) -> dict:
        collection_name = document_id.lower().replace(" ", "_").replace(".", "_")
        self.retriever.delete_document(collection_name)
        return {"message": f"Document '{document_id}' deleted successfully"}
            




