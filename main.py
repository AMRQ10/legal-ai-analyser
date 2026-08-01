from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
import uuid
from typing import cast
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.types import ExceptionHandler

load_dotenv()

import auth.models

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Legal AI Analyser",
    description="AI-powered legal document analysis",
    version="3.0.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded, 
    cast(ExceptionHandler, _rate_limit_exceeded_handler),
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
@limiter.limit("20/minute")
def analyse_clause(request: Request, clause_request: ClauseRequest, current_user: User = Depends(get_current_user)):
    try:
        result = analyser.analyse_clause(clause_request.clause_text)
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
@limiter.limit("20/minute")
def analyse_document(request: Request, document_request: DocumentRequest, current_user: User = Depends(get_current_user)):
    try:
        result = analyser.summarise_document(document_request.document_text)
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
@limiter.limit("20/minute")
def extract_clauses(request: Request, document_request: DocumentRequest, current_user: User = Depends(get_current_user)):
    try:
        result = analyser.extract_key_clauses(document_request.document_text)
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
@limiter.limit("20/minute")
async def analyse_pdf_summary(request: Request, file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
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
@limiter.limit("20/minute")
async def analyse_pdf_clauses(
    request: Request,
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
@limiter.limit("20/minute")
def index_document(request: Request, index_request: IndexRequest, current_user: User = Depends(get_current_user)):
    try:
        result = analyser.index_document(
            index_request.document_text,
            index_request.document_id
        )
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ask")
@limiter.limit("20/minute")
def ask_question(request: Request, question_request: QuestionRequest, current_user : User = Depends(get_current_user)):
    try:
        result = analyser.answer_question(
            question_request.document_id,
            question_request.question
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
@limiter.limit("20/minute")
async def index_pdf(
    request: Request, 
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

task_results = {}

def process_pdf_background(task_id: str, file_bytes: bytes, filename: str):
    """Runs in background after API returns."""
    try:
        task_results[task_id] = {"status": "processing"}

        text = pdf_processor.extract_text(file_bytes)
        result = analyser.summarise_document(text)

        task_results[task_id] = {
            "status": "complete",
            "filename": filename,
            "analysis": result
        }
    except Exception as e:
        task_results[task_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.post("analyse/pdf/async_summary")
@limiter.limit("20/minute")
async def analyse_pdf_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files are accepted")

    file_bytes = await file.read()
    pdf_processor.validate_pdf(file_bytes)

    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "queued"}

    background_tasks.add_task(
        process_pdf_background,
        task_id,
        file_bytes,
        file.filename
    )

    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "Processing started. Poll /tasks/{task_id} for results."
    }

@app.get("/tasks/{task_id}")
def get_task_result(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_results[task_id]
    


if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
