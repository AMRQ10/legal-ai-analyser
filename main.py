from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from analysers.legal_analyser import LegalAnalyser
from auth.router import router as auth_router, get_current_user
from utils.pdf_processor import PDFProcessor
from dotenv import load_dotenv
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from database import engine, get_db, Base
from auth.models import User
from typing import Optional

load_dotenv()

import auth.models

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Legal AI Analyser",
    description="AI-powered legal document analysis",
    version="3.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(auth_router)

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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.0.0"}

@app.post("/analyse/clause")
def analyse_clause(request: ClauseRequest, current_user: User = Depends(get_current_user)):
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
def analyse_document(request: DocumentRequest, current_user: User = Depends(get_current_user)):
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
def extract_clauses(request: DocumentRequest, current_user: User = Depends(get_current_user)):
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
async def analyse_pdf_summary(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if file.filename is None or not file.filename.endswith(".pdf"):
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

@app.post("/analyse/pdf/clauses")
async def analyse_pdf_clauses(
    file: UploadFile = File(...),
    current_user : User = Depends(get_current_user)
):
    if file.filename is None or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    try:
        file_bytes = await file.read()
        pdf_processor.validate_pdf(file_bytes)
        document_text = pdf_processor.extract_text(file_bytes)
        result = analyser.extract_key_clauses(document_text)
        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": pdf_processor.get_page_count(file_bytes),
            "clauses": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/index")
def index_document(request: IndexRequest, current_user: User = Depends(get_current_user)):
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
def ask_question(request: QuestionRequest, current_user : User = Depends(get_current_user)):
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
def list_documents(current_user: User = Depends(get_current_user)):
    documents = analyser.list_indexed_documents()
    return {
        "status": "success",
        "total": len(documents),
        "documents": documents
    }

@app.delete("/rag/documents/{document_id}")
def delete_document(document_id: str, current_user: User = Depends(get_current_user)):
    try:
        result = analyser.delete_indexed_document(document_id)
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/rag/pdf/index")
async def index_pdf(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    if file.filename is None or not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    try:
        file_bytes = await file.read()
        pdf_processor.validate_pdf(file_bytes)
        document_text = pdf_processor.extract_text(file_bytes)

        doc_id = document_id or file.filename.replace(".pdf", "")
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
    
