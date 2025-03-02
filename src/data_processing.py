# import os
# import re
# import json
# import fitz
# import pdfplumber
# import pytesseract
# from PIL import Image
# import io
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from config import config

# class PDFProcessor:
#     def __init__(self, pdf_path):
#         self.pdf_path = pdf_path
#         self.text = ""
#         self.tables = []
#         self.images = []
#         self.formulas = []
#         self.metadata = {}
#         self.page_contents = {}

#     def extract_text(self):
#         """Extract text with page numbers and section headings."""
#         with fitz.open(self.pdf_path) as doc:
#             for page_num, page in enumerate(doc):
#                 page_text = page.get_text("text").strip()
#                 if page_text:
#                     sections = re.findall(r'^\s*(\d+\.\s+.+)$', page_text, re.MULTILINE)
#                     header = f"Page {page_num+1}" + (f" - Sections: {', '.join(sections)}" if sections else "")
#                     self.page_contents[page_num] = f"{header}\n{page_text}"
#             self.text = "\n".join(self.page_contents.values())

#     def extract_tables(self):
#         """Extract tables with page numbers."""
#         with pdfplumber.open(self.pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages):
#                 tables = page.extract_tables()
#                 for table_idx, table in enumerate(tables):
#                     cleaned_table = [row for row in table if any(cell for cell in row if cell)]
#                     if cleaned_table:
#                         table_text = "\n".join(" ".join(str(cell or "") for cell in row) for row in cleaned_table)
#                         self.tables.append({"page": page_num, "table_id": table_idx, "content": table_text})

#     def extract_images_and_formulas(self):
#         """Extract images and detect formulas."""
#         formula_patterns = [
#             r'[=\+\-][^=\+\-\w\s]+[=\+\-]', r'[\(\)]{2,}', r'[∑∫∂√∆∇∏πα-ωΑ-Ω]',
#             r'\$.*?\$', r'\\begin\{equation\}.*?\\end\{equation\}'
#         ]
#         with fitz.open(self.pdf_path) as doc:
#             for page_num, page in enumerate(doc):
#                 for img in page.get_images(full=True):
#                     xref = img[0]
#                     image = Image.open(io.BytesIO(doc.extract_image(xref)["image"]))
#                     text = pytesseract.image_to_string(image)
#                     if any(re.search(pattern, text) for pattern in formula_patterns):
#                         self.formulas.append({"page": page_num, "text": text, "location": "image"})
#                     else:
#                         self.images.append({"page": page_num, "text": text})
#                 page_text = page.get_text("text")
#                 for pattern in formula_patterns:
#                     for formula in re.findall(pattern, page_text, re.DOTALL):
#                         if len(formula) > 3:
#                             self.formulas.append({"page": page_num, "text": formula, "location": "text"})

#     def extract_metadata(self):
#         """Extract metadata with abstract and keywords."""
#         with fitz.open(self.pdf_path) as doc:
#             self.metadata = {k: doc.metadata.get(k, "Unknown") for k in ["title", "author", "creationDate", "modDate"]}
#             self.metadata["page_count"] = doc.page_count
#             first_page = doc[0].get_text("text")
#             abstract_match = re.search(r'(?i)abstract\s*(.+?)\s*(?:keywords|introduction|1\.)', first_page, re.DOTALL)
#             if abstract_match:
#                 self.metadata["abstract"] = abstract_match.group(1).strip()
#             keywords_match = re.search(r'(?i)keywords\s*:?\s*(.+?)\s*(?:introduction|1\.)', first_page, re.DOTALL)
#             if keywords_match:
#                 self.metadata["keywords"] = keywords_match.group(1).strip()

#     def process_pdf(self):
#         """Run all extraction methods and return structured data."""
#         self.extract_text()
#         self.extract_tables()
#         self.extract_images_and_formulas()
#         self.extract_metadata()
#         return {
#             "text": self.text,
#             "tables": [table["content"] for table in self.tables],
#             "images": self.images,
#             "formulas": self.formulas,
#             "metadata": self.metadata,
#             "page_contents": self.page_contents
#         }

# def load_documents():
#     """Load PDF documents using PyPDFLoader."""
#     loader = PyPDFLoader(config.DATA_PATH)
#     return loader.load()

# def split_documents(documents: list[Document]):
#     """Split documents into chunks."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=config.CHUNK_SIZE,
#         chunk_overlap=config.CHUNK_OVERLAP,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)

# def calculate_chunk_metadata(chunks: list[Document]):
#     """Assign chunk IDs and section metadata."""
#     section_markers = {
#         "Abstract": "abstract", "Introduction": "introduction", "Methods": "methods",
#         "Results": "results", "Discussion": "discussion", "References": "references",
#         "Broader Impact": "broader_impact", "Appendices": "appendices"
#     }
#     last_page_id = None
#     current_chunk_index = 0
#     current_section = "other"

#     for chunk in chunks:
#         source = chunk.metadata.get("source", config.DATA_PATH)
#         page = chunk.metadata.get("page", 0)
#         current_page_id = f"{source}:{page}"
#         current_chunk_index = current_chunk_index + 1 if current_page_id == last_page_id else 0
#         chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         content = chunk.page_content
#         for marker, section in section_markers.items():
#             if marker in content:
#                 current_section = section
#                 break
#         chunk.metadata["section"] = current_section
#     return chunks



import os
import re
import json
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config import config
import numpy as np
from sklearn.cluster import DBSCAN

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = ""
        self.tables = []
        self.images = []
        self.formulas = []
        self.metadata = {}
        self.page_contents = {}

    def extract_text(self):
        """Extract text with page numbers and section headings."""
        with fitz.open(self.pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text").strip()
                if page_text:
                    sections = re.findall(r'^\s*(\d+\.\d*\s+.+)$', page_text, re.MULTILINE)  # Enhanced to catch subsections
                    header = f"Page {page_num+1}" + (f" - Sections: {', '.join(sections)}" if sections else "")
                    self.page_contents[page_num] = f"{header}\n{page_text}"
            self.text = "\n".join(self.page_contents.values())

    def extract_tables(self):
        """Extract tables with page numbers."""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    cleaned_table = [row for row in table if any(cell for cell in row if cell)]
                    if cleaned_table:
                        table_text = "\n".join(" ".join(str(cell or "") for cell in row) for row in cleaned_table)
                        self.tables.append({"page": page_num, "table_id": table_idx, "content": table_text})

    def extract_images_and_formulas(self):
        """Extract images and detect formulas."""
        formula_patterns = [
            r'[=\+\-][^=\+\-\w\s]+[=\+\-]', r'[\(\)]{2,}', r'[∑∫∂√∆∇∏πα-ωΑ-Ω]',
            r'\$.*?\$', r'\\begin\{equation\}.*?\\end\{equation\}'
        ]
        with fitz.open(self.pdf_path) as doc:
            for page_num, page in enumerate(doc):
                for img in page.get_images(full=True):
                    xref = img[0]
                    image = Image.open(io.BytesIO(doc.extract_image(xref)["image"]))
                    text = pytesseract.image_to_string(image)
                    if any(re.search(pattern, text) for pattern in formula_patterns):
                        self.formulas.append({"page": page_num, "text": text, "location": "image"})
                    else:
                        self.images.append({"page": page_num, "text": text})
                page_text = page.get_text("text")
                for pattern in formula_patterns:
                    for formula in re.findall(pattern, page_text, re.DOTALL):
                        if len(formula) > 3:
                            self.formulas.append({"page": page_num, "text": formula, "location": "text"})

    def extract_metadata(self):
        """Extract metadata with abstract and keywords."""
        with fitz.open(self.pdf_path) as doc:
            self.metadata = {k: doc.metadata.get(k, "Unknown") for k in ["title", "author", "creationDate", "modDate"]}
            self.metadata["page_count"] = doc.page_count
            first_page = doc[0].get_text("text")
            abstract_match = re.search(r'(?i)abstract\s*(.+?)\s*(?:keywords|introduction|1\.)', first_page, re.DOTALL)
            if abstract_match:
                self.metadata["abstract"] = abstract_match.group(1).strip()
            keywords_match = re.search(r'(?i)keywords\s*:?\s*(.+?)\s*(?:introduction|1\.)', first_page, re.DOTALL)
            if keywords_match:
                self.metadata["keywords"] = keywords_match.group(1).strip()

    def process_pdf(self):
        """Run all extraction methods and return structured data."""
        self.extract_text()
        self.extract_tables()
        self.extract_images_and_formulas()
        self.extract_metadata()
        return {
            "text": self.text,
            "tables": [table["content"] for table in self.tables],
            "images": self.images,
            "formulas": self.formulas,
            "metadata": self.metadata,
            "page_contents": self.page_contents
        }

def load_documents():
    """Load PDF documents using PyPDFLoader."""
    loader = PyPDFLoader(config.DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    """Split documents into semantic and hierarchical chunks with overlap."""
    # Initialize embedding model for semantic chunking
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    # Step 1: Initial split into sentences/paragraphs with hierarchy
    chunks = []
    for doc in documents:
        page_content = doc.page_content
        page_num = doc.metadata.get("page", 0)
        # Split by section headers first (hierarchical)
        sections = re.split(r'(?m)^\s*(\d+\.\d*\s+.+)$', page_content)
        for i, section in enumerate(sections):
            if re.match(r'^\s*\d+\.\d*\s+.+', section):  # Section header
                chunks.append(Document(page_content=section.strip(), metadata={"page": page_num, "type": "header"}))
            else:
                # Split section content into sentences for semantic clustering
                sentences = re.split(r'(?<=[.!?])\s+', section.strip())
                for sent in sentences:
                    if sent.strip():
                        chunks.append(Document(page_content=sent.strip(), metadata={"page": page_num, "type": "content"}))

    # Step 2: Semantic clustering using embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(chunk_texts)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    # Group chunks by cluster
    semantic_chunks = {}
    for idx, label in enumerate(labels):
        if label != -1:  # Ignore noise points
            if label not in semantic_chunks:
                semantic_chunks[label] = []
            semantic_chunks[label].append(chunks[idx])

    # Step 3: Merge clusters into final chunks with overlap
    final_chunks = []
    overlap_size = config.CHUNK_OVERLAP
    for label, cluster in semantic_chunks.items():
        cluster_text = " ".join(chunk.page_content for chunk in cluster)
        cluster_metadata = {"page": cluster[0].metadata["page"], "type": "semantic"}
        
        # Split into chunks with overlap if too large
        if len(cluster_text) > config.CHUNK_SIZE:
            words = cluster_text.split()
            for i in range(0, len(words), config.CHUNK_SIZE - overlap_size):
                chunk_text = " ".join(words[i:i + config.CHUNK_SIZE])
                if chunk_text:
                    final_chunks.append(Document(page_content=chunk_text, metadata=cluster_metadata.copy()))
        else:
            final_chunks.append(Document(page_content=cluster_text, metadata=cluster_metadata))

    # Add tables and formulas as standalone chunks
    processor = PDFProcessor(config.DATA_PATH)
    processed_data = processor.process_pdf()
    for table in processed_data["tables"]:
        final_chunks.append(Document(page_content=table, metadata={"page": -1, "type": "table"}))
    for formula in processed_data["formulas"]:
        final_chunks.append(Document(page_content=formula["text"], metadata={"page": formula["page"], "type": "formula"}))

    return final_chunks

def calculate_chunk_metadata(chunks: list[Document]):
    """Assign chunk IDs and section metadata."""
    section_markers = {
        "Abstract": "abstract", "Introduction": "introduction", "Methods": "methods",
        "Results": "results", "Discussion": "discussion", "References": "references",
        "Broader Impact": "broader_impact", "Appendices": "appendices"
    }
    last_page_id = None
    current_chunk_index = 0
    current_section = "other"

    for chunk in chunks:
        source = chunk.metadata.get("source", config.DATA_PATH)
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"
        current_chunk_index = current_chunk_index + 1 if current_page_id == last_page_id else 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        content = chunk.page_content
        for marker, section in section_markers.items():
            if marker.lower() in content.lower():
                current_section = section
                break
        chunk.metadata["section"] = current_section
    return chunks