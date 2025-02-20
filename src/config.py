import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the RAG chatbot."""
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") 
    CHROMADB_PATH = "D:\\rag-chatbot\DB\chromadb"  
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
    CHUNK_SIZE = 512  # Chunk size for document processing
    CHUNK_OVERLAP = 50

config = Config()
