import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from initialize import initialize_chromadb
from config import config

class EmbedStore:
    # Class to chunk text, generate embeddings, and store in ChromaDB.

    def __init__(self, processed_data_path):
        self.processed_data_path = processed_data_path
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vector_store = initialize_chromadb()

    def load_processed_data(self):
        # Loads processed text, tables, and metadata from JSON file.
        with open(self.processed_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["text"], data["tables"], data["metadata"]

    def chunk_text(self, text):
        # Splits text into smaller chunks for embedding.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False
        )
        return splitter.split_text(text)

    def store_embeddings(self, chunks):
        # Generates embeddings for chunks and stores them in ChromaDB.
        print(f"Processing {len(chunks)} chunks...")
        
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Skip empty chunks
            valid_chunks = [(j, chunk) for j, chunk in enumerate(batch) if chunk.strip()]
            if not valid_chunks:
                continue

            # Prepare valid chunks and metadata
            valid_texts = [chunk for _, chunk in valid_chunks]
            metadatas = [{"chunk_id": f"chunk_{i+j+1}"} 
                        for j, _ in valid_chunks]
            
            # Add the batch to ChromaDB
            self.vector_store.add_texts(
                texts=valid_texts,
                metadatas=metadatas
            )
            # print(f"Processed chunks {i+1} to {i+len(valid_texts)}")
            time.sleep(0.5)  # Small delay between batches

        print(f"Successfully processed {len(chunks)} chunks")

    def process_and_store(self):
        # Main method to load, chunk, embed, and store text.
        text, tables, metadata = self.load_processed_data()
        chunks = self.chunk_text(text)
        self.store_embeddings(chunks)

if __name__ == "__main__":
    processor = EmbedStore("D:\\rag-chatbot\\data\\processed_data.json")
    processor.process_and_store()