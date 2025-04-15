# import chromadb
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
# from langchain_chroma import Chroma
# from chromadb.config import Settings
# from config import config
# from langchain_huggingface import HuggingFacePipeline
# from langchain_openai import ChatOpenAI

# def initialize_chromadb():
#     """Initialize ChromaDB vector store with cosine similarity."""
#     try:
#         embedding_function = HuggingFaceEmbeddings(
#             model_name=config.EMBEDDING_MODEL,
#             encode_kwargs={"normalize_embeddings": True}
#         )
#         chroma_client = Chroma(
#             persist_directory=config.CHROMADB_PATH,
#             embedding_function=embedding_function,
#             collection_name="documents",
#             client_settings=Settings(allow_reset=True, anonymized_telemetry=False)
#         )
#         return chroma_client
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize ChromaDB: {e}")


# def initialize_llm():
#     """Initialize the DeepSeek LLM via OpenRouter API."""
#     try:
#         llm = ChatOpenAI(
#             openai_api_base="https://openrouter.ai/api/v1",  
#             openai_api_key=config.OPENROUTER_API_KEY,        
#             model_name=config.LLM_MODEL,                     
#             temperature=config.LLM_TEMPERATURE,             
#             max_tokens=config.LLM_MAX_TOKENS                 
#         )
#         return llm  
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize LLM: {e}")

# ******************************************************************************


# from llama_cpp import Llama

# def initialize_llm():
#     """Initialize quantized Mistral-7B locally."""
#     llm = Llama(
#         model_path=config.LLM_MODEL,
#         n_ctx=10240,
#         n_threads=6,  
#         n_gpu_layers=0,
#         temperature=config.LLM_TEMPERATURE,
#         max_tokens=config.LLM_MAX_TOKENS,
#         top_p=0.9,
#         repeat_penalty=1.2,
#         verbose=False 
#     )
#     return llm

# from langchain_community.llms import LlamaCpp
# from langchain_core.language_models.llms import LLM

# def initialize_llm() -> LlamaCpp:
#     llm = LlamaCpp(
#         model_path=config.LLM_MODEL,
#         n_ctx=16384,
#         n_threads=6,
#         n_gpu_layers=0,
#         temperature=config.LLM_TEMPERATURE,
#         max_tokens=config.LLM_MAX_TOKENS,
#         top_p=0.9,
#         repeat_penalty=1.2,
#         verbose=True  # change to True to help debugging
#     )
#     return llm



# if __name__ == "__main__":
#     try:
#         llm = initialize_llm()
#         response = llm("What is RAG(Retrieval Augmented Generation)?", max_tokens=50)
#         print(response["choices"][0]["text"])
#     except Exception as e:
#         print(f"Error: {e}")


        
# def initialize_llm():
#     """Initialize Hugging Face LLM for text generation."""
#     try:
#         # llm = HuggingFaceEndpoint(
#         #     repo_id=config.LLM_MODEL,
#         #     task="text-generation",
#         #     huggingfacehub_api_token=config.HUGGINGFACE_API_KEY,
#         #     temperature=config.LLM_TEMPERATURE,
#         #     max_new_tokens=config.LLM_MAX_TOKENS,
#         #     repetition_penalty=1.2  # Reduces repetition
#         # )

#         llm = HuggingFacePipeline.from_model_id(
#             model_id =config.LLM_MODEL,
#             task = 'text-generation',

#             pipeline_kwargs = dict(
#                 temperature = config.LLM_TEMPERATURE,
#                 max_length=2048,
#                 max_new_tokens = 300,
#                 do_sample=True,
#                 repetition_penalty=1.2,
#                 top_p=0.9,
#                 top_k=50
#             )
#         )
#         return ChatHuggingFace(llm=llm)
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize LLM: {e}")



import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from config import config
from langchain_openai import ChatOpenAI

def initialize_chromadb():
    """Initialize ChromaDB vector store with cosine similarity."""
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True}
        )
        chroma_client = Chroma(
            persist_directory=config.CHROMADB_PATH,
            embedding_function=embedding_function,
            collection_name="documents",
            client_settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        return chroma_client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

def initialize_llm():
    """Initialize the DeepSeek LLM via OpenRouter API."""
    try:
        llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=config.OPENROUTER_API_KEY,
            model_name=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")