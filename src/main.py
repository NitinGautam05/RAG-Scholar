import json
import os
from data_processing import PDFProcessor, load_documents, split_documents, calculate_chunk_metadata
from embed_store import add_to_chroma
from initialize import initialize_chromadb, initialize_llm
from config import config
from chat import ChatHandler
import camelot

def main():
    # Process PDF and save structured data
    processor = PDFProcessor(config.DATA_PATH)
    processed_data = processor.process_pdf()
    output_path = os.path.join(os.path.dirname(config.DATA_PATH), "extracted_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)
    print(f"PDF processing completed! Data saved at {output_path}")

    # Load, split, and embed documents
    documents = load_documents()
    chunks = split_documents(documents)
    chunks_with_metadata = calculate_chunk_metadata(chunks)
    db = add_to_chroma(chunks_with_metadata)

    # Initialize LLM
    llm = initialize_llm()


    chat_handler = ChatHandler(db, llm, config)
    chat_handler.run()

if __name__ == "__main__":
    main()

