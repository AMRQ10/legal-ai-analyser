import anthropic
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

class LegalAnalyser:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.model = "llama-3.3-70b-versatile"
        self.system_prompt= """You are an expert legal analyst with deep knowledge of contract law, corporate law, and regulatory compliance.

CRITICAL INSTRUCTION: You must ALWAYS respond with valid JSON only. 
No preamble, no explanation outside the JSON, no markdown code blocks.
Just raw, valid JSON matching the exact schema requested.

When analysing legal text you:
- Identify key obligations and rights for each party
- Flag potentially problematic or unusual clauses
- Highlight missing standard protections
- Explain complex legal language in plain English
- Rate risk level as LOW, MEDIUM, or HIGH with justification
- Assign a numerical risk score from 1-10"""

    def _call_api(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)

    def analyse_clause(self, clause_text: str) -> dict:
        if not clause_text or not clause_text.strip():
            raise ValueError("Clause text cannot be empty")

        clause_text = clause_text.replace('\r', ' ').replace('\n', ' ').strip()

        prompt = f"""Analyze this contract clause and return a JSON object
        with exactly this structure:
        {{
            "plain_english": "explanation in simple terms",
            "obligations": ["obligation 1", "obligation 2"],
            "risks": ["risk 1", "risk 2"],
            "risk_rating": "LOW or MEDIUM or HIGH",
            "risk_score": 7,
            "recommendations": ["recommendation 1", "recommendation 2"],
            "clause_type": "e.g. Non-Compete, Confidentiality, Termination"
        }}
        
        CLAUSE:
        {clause_text}"""

        try:
            raw = self._call_api(prompt, max_tokens=1024)
            return self._parse_json(raw)
        
        except json.JSONDecodeError:
            raise ValueError("Model returned invalid JSON. Try again.")
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")
        
    def summarise_document(self, document_text: str) -> dict:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")
        
        document_text = document_text.replace('\r', ' ').replace('\n', ' ').strip()

        prompt = f"""Analyze this legal document and return a JSON object with exactly this structure:
        {{
            "document_type": "e.g. NDA, Employment Contract, Lease Agreement",
            "parties": [
                {{"name": "Party A", "role": "e.g. Employer"}},
                {{"name": "Party B", "role": "e.g Employee"}}
            ],
            "purpose": "one sentence description",
            "key_terms": ["term 1", "term 2", "term 3"],
            "obligations": {{
                "party_a": ["obligation 1", "obligations 2"],
                "party_b": ["obligation 1", "obligation 2"]
            }},
            "concerns": ["concern 1", "concern 2"],
            "missing_protections": ["missing item 1", "missing item 2"],
            "overall_risk_rating": "LOW or MEDIUM or HIGH",
            "overall_risk_score": 6,
            "summary": "2-3 sentences plain English summary"
        }}

        DOCUMENT:
        {document_text}"""

        try:
            raw = self._call_api(prompt, max_tokens=2048)
            return self._parse_json(raw)

        except json.JSONDecodeError:
            raise ValueError("Model returned invalid JSON. Try again.")
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")
    
    def extract_key_clauses(self, document_text: str) -> dict:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")

        prompt = f"""Extract and categorise all key clauses from this
        legal document and return a JSON object with exactly this structure:
        {{
            "total_clauses_found": 5,
            "high_risk_count": 1, 
            "medium_risk_count": 2,
            "low_risk_count": 2,
            "clauses": [
                {{
                    "clause_type": "e.g. Non-Compete",
                    "verbatim_text": "exact text from document",
                    "plain_english": "simple explanation",
                    "risk_rating": "HIGH",
                    "risk_score": 8,
                    "concerns": ["concern 1"]

                }}
            ]
        }}

        Order clauses by risk_score descending (highest risk first).

        DOCUMENT:
        {document_text}"""

        try:
            raw = self._call_api(prompt, max_tokens=2048)
            return self._parse_json(raw)

        except json.JSONDecodeError:
            raise ValueError("Model returned invalid JSON. Try again.")
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {e}")




