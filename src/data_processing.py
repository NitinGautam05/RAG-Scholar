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
import pandas as pd
from sklearn.cluster import DBSCAN
import camelot
from OcrToTableTool import OcrTableExtractor
from table import TableExtractor

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = ""
        self.tables = []
        self.images = []
        self.formulas = []
        self.metadata = {}
        self.page_contents = {}
        self.page_table_captions = {}

    def extract_text(self):
        """Extract text and table captions."""
        table_caption_pattern = r'(Table\s+\d+[\s:].*?)(?=\n\s*\d+\.|\Z)'  # Matches "Table X: Description"
        with fitz.open(self.pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text").strip()
                if page_text:
                    # Extract sections
                    sections = re.findall(r'^\s*(\d+\.\d*\s+.+)$', page_text, re.MULTILINE)
                    header = f"Page {page_num+1}" + (f" - Sections: {', '.join(sections)}" if sections else "")
                    self.page_contents[page_num] = f"{header}\n{page_text}"
                    
                    # Extract table captions
                    captions = re.findall(table_caption_pattern, page_text, re.DOTALL | re.IGNORECASE)
                    self.page_table_captions[page_num] = [cap.strip() for cap in captions]

        self.text = "\n".join(self.page_contents.values())

    def extract_tables(self):
        """Extract tables from the PDF using TableExtractor and OcrTableExtractor."""
        # Step 1: Extract table images
        table_extractor = TableExtractor(self.pdf_path, output_dir="D:/rag-chatbot/data")
        table_extractor.run()

        # Step 2: Process table images with OCR to generate JSON outputs
        ocr_extractor = OcrTableExtractor(
            image_dir=table_extractor.table_output_dir,
            output_dir="D:/rag-chatbot/data/output"
        )
        ocr_extractor.process_all_tables()

        # Step 3: Collect extracted tables from JSON files
        self.tables = []
        output_dir = ocr_extractor.output_dir
        for json_file in os.listdir(output_dir):
            if json_file.endswith("_extracted.json"):
                json_path = os.path.join(output_dir, json_file)
                try:
                    # Extract page and table numbers from filename
                    match = re.search(r'page_(\d+)_table_(\d+)_', json_file)
                    if match:
                        page_num = int(match.group(1)) - 1  # 0-based index
                        table_num = int(match.group(2)) - 1  # 0-based index
                    else:
                        table_num = 0
                        page_num = 0

                    # Get captions for the page
                    captions = self.page_table_captions.get(page_num, [])
                    caption = (
                        captions[table_num] 
                        if table_num < len(captions) 
                        else f"Table {table_num + 1}"
                    )

                    # Read table content from JSON
                    with open(json_path, "r", encoding="utf-8") as f:
                        table_data = json.load(f)
                    # Convert JSON back to DataFrame for a text representation
                    df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
                    table_content = df.to_string(index=False)
                    
                    self.tables.append({
                        "page": page_num,
                        "table_id": table_num + 1,  # 1-based for user-facing
                        "caption": caption,
                        "content": table_content
                    })
                except Exception as e:
                    print(f"Error reading table from {json_path}: {e}")


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
            # "tables": [table["content"] for table in self.tables],
            "tables": [ 
                {
                    "id": f"page-{table['page']}-table-{table['table_id']}",
                    "caption": table["caption"],
                    "content": table["content"]
                } 
                for table in self.tables
            ],
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
        final_chunks.append(Document(page_content=f"Table {table['id']} - {table['caption']}\n{table['content']}",metadata={"page": -1, "type": "table"}))
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

if __name__ == "__main__":
    processor = PDFProcessor(config.DATA_PATH)
    result = processor.process_pdf()

    # Define output file path
    output_json_path = os.path.join("D:/rag-chatbot/data", "extracted_data.json")  

    # Save JSON output to a file
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False, default=str)

    print(f"JSON file saved at: {output_json_path}")

