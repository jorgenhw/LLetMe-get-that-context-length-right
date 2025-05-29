import re
import pdfplumber
from typing import Tuple, List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import tiktoken
from .data_structures import Document

class DocumentProcessor:
    def __init__(self, should_anonymize: bool = True):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.should_anonymize = should_anonymize
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extract text and separate pages from PDF"""
        full_text = ""
        pages = [] # for tracking the page number through the entire process...
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Clean formatting
                    page_text = self._clean_text(page_text)
                    pages.append(page_text)
                    full_text += f"<PAGE_BREAK>{i+1}</PAGE_BREAK>\n{page_text}\n"
        
        return full_text, pages
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove headers/footers (simple pattern)
        text = re.sub(r'^.*Page \d+ of \d+.*$', '', text, flags=re.MULTILINE)
        # Fix broken lines
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        return text.strip()
    
    def anonymize_document(self, text: str) -> str:
        """Anonymize sensitive information"""
        if not self.should_anonymize:
            return text
        results = self.analyzer.analyze(text=text, language='en')
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def create_document(self, pdf_path: str, doc_id: str) -> Document:
        """Create a Document object from a PDF file"""
        
        # Extract text from PDF
        full_text, pages = self.extract_text_from_pdf(pdf_path)
        
        # Anonymize if needed
        if self.should_anonymize:
            processed_text = self.anonymize_document(full_text)
            processed_pages = [self.anonymize_document(page) for page in pages]
        else:
            processed_text = full_text
            processed_pages = pages
        
        # Create document
        document = Document(
            id=doc_id,
            content=processed_text,
            pages=processed_pages,
            length_tokens=self.count_tokens(processed_text),
            length_pages=len(pages),
            needles=[]
        )
        
        return document
