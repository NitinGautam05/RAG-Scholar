import fitz  # PyMuPDF
import pdfplumber
import os
import json

class PDFProcessor:
    # Class for extracting text, tables, and metadata from PDFs.

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = ""
        self.tables = []
        self.metadata = {}

    def extract_text(self):
        # Extracts text from a PDF using PyMuPDF.
        doc = fitz.open(self.pdf_path)
        extracted_text = []
        for page in doc:
            extracted_text.append(page.get_text("text"))
        self.text = "\n".join(extracted_text)

    def extract_tables(self):
        # Extracts tables from a PDF using pdfplumber.
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    self.tables.extend(tables)

    def extract_metadata(self):
        # Extracts metadata (title, author, etc.) from PDF.
        doc = fitz.open(self.pdf_path)
        metadata = doc.metadata
        self.metadata = {
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "creation_date": metadata.get("creationDate", "Unknown"),
            "modification_date": metadata.get("modDate", "Unknown"),
        }

    def process_pdf(self):
        # Runs all extraction methods.
        self.extract_text()
        self.extract_tables()
        self.extract_metadata()

        return {
            "text": self.text,
            "tables": self.tables,
            "metadata": self.metadata
        }

if __name__ == "__main__":
    # Test with a sample PDF
    pdf_path = "D:\\rag-chatbot\data\\rag_paper.pdf"
    processor = PDFProcessor(pdf_path)
    processed_data = processor.process_pdf()
    
    # Save extracted data as JSON
    output_path = os.path.join("D:\\", "rag-chatbot", "data", "processed_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    print(f"✅ PDF processing completed! Data saved at {output_path}")
