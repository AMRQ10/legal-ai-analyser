# Legal AI Analyser

An AI-powered legal document analysis tool built with FastAPI and Large Language Models. Analyses contracts, flags risks, extracts clauses, and answers specific legal questions using RAG (Retrieval Augmented Generation).

![Legal AI Analyser Interface](screenshot.png)

## Live Features

- **Clause Analysis** — paste any contract clause and get a structured 
  risk assessment with plain English explanation, obligations, and recommendations
- **Document Summary** — full contract analysis identifying parties, key terms, 
  concerns, and missing protections
- **Clause Extraction** — extracts and ranks all clauses by risk level (HIGH/MEDIUM/LOW)
- **Document Q&A** — ask specific questions about any contract using RAG, 
  getting precise answers from relevant sections only

## Tech Stack

- **Backend** — Python, FastAPI
- **AI Layer** — Groq LLM API (Llama 3.3 70B), sentence-transformers
- **RAG Pipeline** — ChromaDB vector database, all-MiniLM-L6-v2 embeddings
- **Document Processing** — PyPDF2 for PDF text extraction
- **Frontend** — HTML, CSS, JavaScript served via Jinja2 templates
- **Database** — PostgreSQL, SQLAlchemy

## Project Structure

legal-ai-analyser/

├── analysers/

│   └── legal_analyser.py    # LLM analysis and RAG pipeline

├── rag/

│   ├── chunker.py           # Document chunking with overlap

│   ├── embedder.py          # Sentence transformer embeddings

│   └── retriever.py         # ChromaDB vector storage and search

├── utils/

│   └── pdf_processor.py     # PDF validation and text extraction

├── static/

│   └── style.css            # Frontend styles

├── templates/

│   └── index.html           # Web interface

└── main.py                  # FastAPI application and endpoints

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Web interface |
| GET | /health | Health check |
| POST | /analyse/clause | Analyse a contract clause |
| POST | /analyse/document | Summarise a full document |
| POST | /analyse/extract-clauses | Extract and rank all clauses |
| POST | /analyse/pdf/summary | Upload PDF for summary |
| POST | /analyse/pdf/clauses | Upload PDF for clause extraction |
| POST | /rag/index | Index a document for Q&A |
| POST | /rag/ask | Ask a question about a document |
| GET | /rag/documents | List all indexed documents |
| DELETE | /rag/documents/{id} | Delete an indexed document |
| POST | /rag/pdf/index | Upload and index a PDF |

## Local Setup

```bash
git clone https://github.com/AMRQ10/legal-ai-analyser
cd legal-ai-analyser
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=groq_api_key_here

APP_NAME=Legal AI Analyser

Run the server:
uvicorn main:app --reload


Open `http://127.0.0.1:8000`

## How the RAG Pipeline Works

1. Document text is split into overlapping chunks of 500 characters
2. Each chunk is converted to a vector embedding using sentence-transformers
3. Embeddings are stored in ChromaDB with metadata
4. When a question is asked, it is embedded and compared against stored chunks
5. The 5 most semantically similar chunks are retrieved
6. The LLM answers the question using only those relevant chunks

This allows accurate, grounded answers from documents of any length without 
exceeding model context limits.

## Example Analysis Output

**Clause Analysis — Non-Compete Clause:**

```json
{
  "clause_type": "Non-Compete",
  "risk_rating": "HIGH",
  "risk_score": 8,
  "plain_english": "You cannot work for any competitor for 2 years after leaving",
  "obligations": ["Cannot join competing firms", "Geographic restriction applies"],
  "risks": ["Overly broad scope", "Long duration", "No carve-outs"],
  "recommendations": ["Narrow geographic scope", "Reduce to 6-12 months"]
}
```

## Author

Built as part of a structured path toward specialising in AI-powered legal technology.

[GitHub](https://github.com/AMRQ10)

