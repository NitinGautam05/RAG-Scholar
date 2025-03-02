import json
import os
from data_processing import PDFProcessor, load_documents, split_documents, calculate_chunk_metadata
from embed_store import add_to_chroma
from initialize import initialize_chromadb, initialize_llm
from config import config

def main():
    # Process PDF and save structured data
    processor = PDFProcessor(config.DATA_PATH)
    processed_data = processor.process_pdf()
    output_path = os.path.join(os.path.dirname(config.DATA_PATH), "processed_data.json")
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

    # Query loop
    while True:
        query = input("\nEnter your query about the paper (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        docs = db.similarity_search(query, k=config.TOP_K)
        unique_docs = list({doc.page_content: doc for doc in docs}.values())
        context = "\n\n".join(doc.page_content for doc in unique_docs[:config.TOP_K])
        prompt = (
            "<|user|>\n"
            "Using only the following context from a research paper, provide a concise and accurate response to the question. "
            "Do not introduce information outside the context.\n"
            f"Context:\n{context}\n"
            f"Question: {query}\n"
            "<|assistant|>\n"
        )
        response = llm.invoke(prompt)
        print("\nResponse:\n", response.content)

if __name__ == "__main__":
    main()