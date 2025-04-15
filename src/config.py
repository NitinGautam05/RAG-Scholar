# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Config:
    
#     # API Keys and paths
#     HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
#     OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
#     CHROMADB_PATH = os.getenv("CHROMADB_PATH", "D:/rag-chatbot/DB/chromadb")
#     # DATA_PATH = os.getenv("DATA_PATH", "D:/rag-chatbot/data/CAG.pdf")
#     DATA_PATH = None 

#     # Embedding and chunking settings
#     EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
#     CHUNK_SIZE = 1500
#     CHUNK_OVERLAP = 300
#     BATCH_SIZE = 5
#     TOP_K = 5

#     # LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     # LLM_MODEL = "D:\\rag-chatbot\model\mistral-7b-instruct-v0.1.Q4_K_M.gguf"  
#     LLM_MODEL = "deepseek/deepseek-r1-distill-llama-70b:free"
#     LLM_TEMPERATURE = 0.7
#     LLM_MAX_TOKENS = 1000

# config = Config()

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Paths (relative for cloud compatibility)
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chromadb")
    DATA_PATH = None  # Will be set dynamically via file upload
    
    # Embedding and chunking settings
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    BATCH_SIZE = 5
    TOP_K = 5
    
    # LLM settings
    LLM_MODEL = "deepseek/deepseek-r1-distill-llama-70b:free"
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 1000

config = Config()