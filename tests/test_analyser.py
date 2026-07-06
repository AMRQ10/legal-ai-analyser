import pytest
from analysers.legal_analyser import LegalAnalyser
from utils.pdf_processor import PDFProcessor

@pytest.fixture
def analyser():
    return LegalAnalyser()

def test_analyse_clause_rejects_empty_string(analyser):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyser.analyse_clause("")

def test_analyse_clause_rejects_whitespace_only(analyser):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyser.analyse_clause("  ")

def test_summarise_document_rejects_empty_string(analyser):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyser.summarise_document("")

def test_extract_clauses_rejects_empty_string(analyser):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyser.extract_key_clauses("")

@pytest.fixture
def pdf_processor():
    return PDFProcessor()

def test_validate_pdf_rejects_empty_file(pdf_processor):
    with pytest.raises(ValueError, match="empty"):
        pdf_processor.validate_pdf(b"")

def test_validate_pdf_rejects_oversized_file(pdf_processor):
    huge_file = b"0" * (11 * 1024 * 1024)  # 11 MB, over the 10 MB limit
    with pytest.raises(ValueError, match="too large"):
        pdf_processor.validate_pdf(huge_file)


