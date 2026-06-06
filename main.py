from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from analysers.legal_analyser import LegalAnalyser
from utils.pdf_processor import PDFProcessor
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="Legal AI Analyser",
    description="AI-powered legal document analysis",
    version="2.0.0"
)

analyser = LegalAnalyser()
pdf_processor = PDFProcessor()

# --- Text-based endpoints (existing) ---

class ClauseRequest(BaseModel):
    clause_text: str

class DocumentRequest(BaseModel):
    document_text: str

class IndexRequest(BaseModel):
    document_text: str
    document_id: str

class QuestionRequest(BaseModel):
    document_id: str
    question: str

@app.get("/")
def root():
    return {
        "message": "Legal AI Analyser is running",
        "endpoints": [
            "/analyse/clause",
            "/analyse/document",
            "/analyse/extract-clauses",
            "/analyse/pdf/summary",
            "/analyse/pdf/clauses"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "llama-3.3-70b-versatile"}

@app.post("/analyse/clause")
def analyse_clause(request: ClauseRequest):
    try:
        result = analyser.analyse_clause(request.clause_text)
        return {
            "status": "success",
            "analysis": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse/document")
def analyse_document(request: DocumentRequest):
    try:
        result = analyser.summarise_document(request.document_text)
        return {"status": "success", "analysis": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse/extract-clauses")
def extract_clauses(request: DocumentRequest):
    try:
        result = analyser.extract_key_clauses(request.document_text)
        return {
            "status": "success",
            "clauses": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyse/pdf/summary")
async def analyse_pdf_summary(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        file_bytes = await file.read()
        pdf_processor.validate_pdf(file_bytes)
        document_text = pdf_processor.extract_text(file_bytes)
        result = analyser.summarise_document(document_text)

        return {
            "status": "success",
            "file_name": file.filename,
            "pages_processed": pdf_processor.get_page_count(file_bytes),
            "clauses": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/index")
def index_document(request: IndexRequest):
    try:
        result = analyser.index_document(
            request.document_text,
            request.document_id
        )
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ask")
def ask_question(request: QuestionRequest):
    try:
        result = analyser.answer_question(
            request.document_id,
            request.question
        )
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/documents")
def list_documents():
    documents = analyser.list_indexed_documents()
    return {
        "status": "success",
        "total": len(documents),
        "documents": documents
    }

@app.delete("/rag/documents/{document_id}")
def delete_document(document_id: str):
    try:
        result = analyser.delete_indexed_documet(document_id)
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/rag/pdf/index")
async def index_pdf(
    file: UploadFile = File(...),
    document_id: str = None
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    try:
        file_bytes = await file.read()
        pdf_processor.validate_pdf(file_bytes)
        document_text = pdf_processor.extract_text(file_bytes)

        doc_id = document_id or file.filename.replace(".pdf")
        result = analyser.index_document(document_text, doc_id)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": pdf_processor.get_page_count(file_bytes),
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
