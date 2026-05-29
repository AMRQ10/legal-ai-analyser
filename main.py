from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from analysers.legal_analyser import LegalAnalyser
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="Legal AI Analyser",
    description="AI-powered legal document analysis using Gemini",
    version="gemini-2.0-flash"
)

analyser = LegalAnalyser()

class ClauseRequest(BaseModel):
    clause_text: str

class DocumentRequest(BaseModel):
    document_text: str

@app.get("/")
def root():
    return {
        "message": "Legal AI Analyser is running",
        "endpoints": [
            "/analyse/clause",
            "/analyse/document",
            "/analyse/extract-clauses"
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

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
