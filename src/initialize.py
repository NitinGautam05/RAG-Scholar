import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from config import config

def initialize_chromadb():

    # Initialize and return a ChromaDB vector store with Hugging Face embeddings.
    embedding_function = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=config.CHROMADB_PATH,
        embedding_function=embedding_function,
        collection_name="documents"
    )

def initialize_llm():

    # Initialize and return a Hugging Face LLM for text generation.
    llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        huggingfacehub_api_token=config.HUGGINGFACE_API_KEY
    )
    return ChatHuggingFace(llm=llm)

