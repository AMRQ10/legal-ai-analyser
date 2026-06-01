import PyPDF2
import io

class PDFProcessor:

    def extract_text(self, file_bytes: bytes) -> str:
        """
        Extracts all text from a PDF file given its raw bytes.
        Returns the extracted text as a single string.
        """
        try:
            pdf_file = io.BytesIO(file_bytes)
            reader = PyPDF2.PdfReader(pdf_file)

            if len(reader.pages) == 0:
                raise ValueError("PDF has no pages")

            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text

            if not text.strip():
                raise ValueError(
                    "Could not extract text from PDF. "
                    "The file may be scanned or image-based."
                )

            return text.strip()

        except PyPDF2.errors.PdfReadError:
            raise ValueError("Invalid or corrupted PDF file")
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {e}")

    def get_page_count(self, file_bytes: bytes) -> int:
        pdf_file = io.BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        return len(reader.pages)
    
    def validate_pdf(self, file_bytes: bytes, max_pages: int = 50) -> None:
        """
        Validates PDF before processing.
        Raises ValueError if validation fails.
        """
        if len(file_bytes) == 0:
            raise ValueError("File is empty")

        if len(file_bytes) > 10 * 1024 * 1024: # 10 MB limit
            raise ValueError("File too large. Maximum size is 10 MB")
        
        page_count = self.get_page_count(file_bytes)

        if page_count > max_pages:
            raise ValueError(
                f"Document is too long. Maximum {max_pages} pages,"
                f"your document has {page_count} pages."
            )