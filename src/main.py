import os
import json
from data_processing import PDFProcessor
from embed_store import EmbedStore

def main():
    # Main function to process PDFs and store embeddings.
    pdf_path = "D:\\rag-chatbot\data\\rag_paper.pdf"
    
    # Step 1: Extract text, tables, and metadata
    print("🔄 Processing PDF...")
    processor = PDFProcessor(pdf_path)
    processed_data = processor.process_pdf()

    # Save processed data
    output_path = "D:\\rag-chatbot\data\\processed_data.json"
    os.makedirs("data", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    print(f"✅ PDF processed and saved at {output_path}")

    # Step 2: Chunk, embed, and store in ChromaDB
    embedder = EmbedStore(output_path)
    embedder.process_and_store()

    print("🚀 Data ingestion pipeline completed!")

if __name__ == "__main__":
    main()
