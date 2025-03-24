import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    # API Keys and paths
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "D:/rag-chatbot/DB/chromadb")
    DATA_PATH = os.getenv("DATA_PATH", "D:/rag-chatbot/data/CAG.pdf")

    # Embedding and chunking settings
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    BATCH_SIZE = 5
    TOP_K = 5

    # LLM settings
    # LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LLM_MODEL = "deepseek/deepseek-r1-distill-llama-70b:free"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 1000

    def __init__(self):
        """Validate required environment variables."""
        if not self.HUGGINGFACE_API_KEY:
            raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found in .env")
        if not self.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

config = Config()