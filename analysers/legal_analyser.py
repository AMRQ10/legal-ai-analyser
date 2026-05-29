from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class LegalAnalyser:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.model = "llama-3.3-70b-versatile"
        self.system_prompt="""You are an expert legal analyst with deep knowledge of contract law, corporate law, and regulatory compliance.

            When analysing legal text you:
            - Identify key obligations and rights for each party
            - Flag potentially problematic or unusual clauses
            - Highlight missing standard protections
            - Explain complex legal language in plain English
            - Rate risk level as LOW, MEDIUM, or HIGH with justification

            Always structure your response clearly with labeled sections.
            Be precise, objective, and flag ambiguities explicitly."""

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

    def analyse_clause(self, clause_text: str) -> str:
        if not clause_text or not clause_text.strip():
            raise ValueError("Clause text cannot be empty")

        clause_text = clause_text.replace('\r', ' ').replace('\n', ' ').strip()

        prompt = f"""Analyze the following contract clause and provide:
        1. PLAIN ENGLISH EXPLANATION
        2. KEY OBLIGATIONS CREATED
        3. POTENTIAL RISKS OR ISSUES
        4. RISK TAKING (LOW, MEDIUM, HIGH) with justification
        5. RECOMMENDED MODIFICARTIONS if needed

        CLAUSE:
        {clause_text}"""

        try:
            return self._call_api(prompt)
        except Exception as e:
            raise RuntimeError(f"API error: {e}")
        
    def summarise_document(self, document_text: str) -> str:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")
        
        document_text = document_text.replace('\r', ' ').replace('\n', ' ').strip()

        prompt = f"""Analyze this legal document and provide:
        1. DOCUMENT TYPE and PURPOSE
        2. PARTIES INVOLVED and their roles
        3. KEY TERMS AND CONDITIONS (bullet points)
        4. NOTABLE OBLIGATIONS for each party
        5. POTENTIAL CONCERNS for unusual provisions
        6. OVERALL RISK ASSESSMENT(LOW/MEDIUM/HIGH)

        DOCUMENT:
        {document_text}"""

        try:
            return self._call_api(prompt, max_tokens=2048)

        except Exception as e:
            raise RuntimeError(f"API error: {e}")
    
    def extract_key_clauses(self, document_text: str) -> str:
        if not document_text or not document_text.strip():
            raise ValueError("Document text cannot be empty")

        prompt = f"""Extract and categorise all key clauses from this
        legal document. For each clause provide:
        - CLAUSE TYPE (e.g. Termination, Liability, Confidentiality)
        - CLAUSE TEXT (verbatim)
        - PLAIN ENGLISH SUMMARY
        - RISK LEVEL (LOW/MEDIUM/HIGH)

        Format as a structured list ordered by risk level, highest first.

        DOCUMENT:
        {document_text}"""

        try:
            return self._call_api(prompt, max_tokens=2048)

        except Exception as e:
            raise RuntimeError(f"API error {e}")




