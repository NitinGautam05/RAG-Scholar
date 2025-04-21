# import streamlit as st
# import os
# import json
# import tempfile
# import shutil
# import time
# import gc  # Added for garbage collection
# from config import config
# from data_processing import PDFProcessor, load_documents, split_documents, calculate_chunk_metadata
# from embed_store import add_to_chroma, robust_delete_chroma_db
# from initialize import initialize_chromadb, initialize_llm
# from chat import ChatHandler

# st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
# st.title("📄💬 RAG Chatbot for Research Papers")

# # --- Sidebar ---
# with st.sidebar:
#     st.header("🧠 Chat Options")

#     if "history" in st.session_state and st.session_state.history:
#         st.subheader("📜 Chat History")
#         for i, (q, a) in enumerate(st.session_state.history[::-1]):
#             st.markdown(f"**Q{i+1}:** {q}")
#             st.markdown(f"🟢 {a}")

#         st.markdown("---")
#         if st.download_button("📥 Download Chat", "\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state.history]), file_name="chat_history.txt"):
#             st.toast("Chat history downloaded!")

#     if st.button("🆕 New Chat"):
#         # Clean up session state in proper order
#         for key in ["chatbot", "db", "llm", "history", "last_uploaded_name"]:
#             if key in st.session_state:
#                 del st.session_state[key]
        
#         # Force cleanup of resources
#         gc.collect()
        
#         # Handle Chroma DB deletion
#         if robust_delete_chroma_db():
#             st.success("✅ Chroma DB successfully reset")
#         else:
#             st.error("❌ Failed to reset Chroma DB - please try again")
        
#         st.rerun()

# # --- PDF Upload ---
# uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if uploaded_file:
#     if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_file.read())
#             uploaded_path = tmp_file.name

#         config.DATA_PATH = uploaded_path
#         processor = PDFProcessor(config.DATA_PATH)

#         with st.spinner("Processing PDF..."):
#             processed_data = processor.process_pdf()
#             output_path = os.path.join(os.path.dirname(config.DATA_PATH), "extracted_data.json")

#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             with open(output_path, "w", encoding="utf-8") as f:
#                 json.dump(processed_data, f, indent=4)

#             documents = load_documents()
#             chunks = split_documents(documents, processed_data=processed_data)
#             chunks_with_metadata = calculate_chunk_metadata(chunks)
            
#             # Ensure Chroma directory is clean before creating new instance
#             if os.path.exists("D:/rag-chatbot/DB/chromadb"):
#                 robust_delete_chroma_db()
            
#             db = add_to_chroma(chunks_with_metadata)

#         st.session_state.db = db
#         if "llm" not in st.session_state:
#             st.session_state.llm = initialize_llm()

#         st.session_state.chatbot = ChatHandler(st.session_state.db, st.session_state.llm, config)
#         st.session_state.history = []
#         st.session_state.last_uploaded_name = uploaded_file.name

#         st.success("✅ PDF processed! You can now chat below.")

# # --- Chat Input & Display ---
# if "chatbot" in st.session_state:
#     question = st.text_input("Ask a question:")

#     if question:
#         response = st.session_state.chatbot.generate_response(question)
#         st.session_state.chatbot.update_history(question, response)
#         st.session_state.history.append((question, response))

# if "history" in st.session_state:
#     for user_q, bot_a in st.session_state.history:
#         st.markdown(f"**You:** {user_q}")
#         st.markdown(f"**Bot:** {bot_a}")

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import streamlit as st
# import os
# import json
# import tempfile
# import shutil
# from config import config
# from data_processing import PDFProcessor, load_documents, split_documents, calculate_chunk_metadata
# from embed_store import add_to_chroma, robust_delete_chroma_db
# from initialize import initialize_chromadb, initialize_llm
# from chat import ChatHandler

# st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
# st.title("📄💬 RAG Chatbot for Research Papers")

# # Initialize session state
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "last_uploaded_name" not in st.session_state:
#     st.session_state.last_uploaded_name = None

# # Sidebar
# with st.sidebar:
#     st.header("🧠 Chat Options")
#     if st.session_state.history:
#         st.subheader("📜 Chat History")
#         for i, (q, a) in enumerate(st.session_state.history[::-1]):
#             st.markdown(f"**Q{i+1}:** {q}")
#             st.markdown(f"🟢 {a}")
#         st.markdown("---")
#         chat_data = "\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state.history])
#         st.download_button("📥 Download Chat", chat_data, file_name="chat_history.txt")
    
#     if st.button("🆕 New Chat"):
#         robust_delete_chroma_db()
#         for key in ["chatbot", "db", "llm", "history", "last_uploaded_name"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.session_state.history = []
#         st.success("✅ Session reset")
#         st.rerun()

# # PDF Upload
# uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if uploaded_file:
#     if st.session_state.last_uploaded_name != uploaded_file.name:
#         with st.spinner("Processing PDF..."):
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 config.DATA_PATH = tmp_file.name

#             try:
#                 # Process PDF
#                 processor = PDFProcessor(config.DATA_PATH)
#                 processed_data = processor.process_pdf()
                
#                 # Save extracted data
#                 output_dir = "./data"
#                 os.makedirs(output_dir, exist_ok=True)
#                 output_path = os.path.join(output_dir, "extracted_data.json")
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     json.dump(processed_data, f, indent=4)

#                 # Load and chunk documents
#                 documents = load_documents()
#                 chunks = split_documents(documents, processed_data=processed_data)
#                 chunks_with_metadata = calculate_chunk_metadata(chunks)

#                 # Initialize ChromaDB
#                 robust_delete_chroma_db()
#                 db = add_to_chroma(chunks_with_metadata)

#                 # Initialize LLM
#                 llm = initialize_llm()

#                 # Setup chatbot
#                 st.session_state.db = db
#                 st.session_state.llm = llm
#                 st.session_state.chatbot = ChatHandler(db, llm, config)
#                 st.session_state.history = []
#                 st.session_state.last_uploaded_name = uploaded_file.name

#                 st.success("✅ PDF processed! You can now chat.")
#             except Exception as e:
#                 st.error(f"❌ Error processing PDF: {e}")
#             finally:
#                 if config.DATA_PATH and os.path.exists(config.DATA_PATH):
#                     os.unlink(config.DATA_PATH)

# # Chat Interface
# if "chatbot" in st.session_state:
#     question = st.text_input("Ask a question:")
#     if question:
#         with st.spinner("Generating response..."):
#             try:
#                 response = st.session_state.chatbot.generate_response(question)
#                 st.session_state.chatbot.update_history(question, response)
#                 st.session_state.history.append((question, response))
#             except Exception as e:
#                 st.error(f"❌ Error generating response: {e}")

#     for user_q, bot_a in st.session_state.history:
#         st.markdown(f"**You:** {user_q}")
#         st.markdown(f"**Bot:** {bot_a}")

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import streamlit as st
# import os
# import json
# import tempfile
# import shutil
# from config import config
# from data_processing import PDFProcessor, load_documents, split_documents, calculate_chunk_metadata
# from embed_store import add_to_chroma, robust_delete_chroma_db
# from initialize import initialize_chromadb, initialize_llm
# from chat import ChatHandler

# st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
# st.title("📄💬 RAG Chatbot for Research Papers")

# # Initialize session state
# if "history" not in st.session_state:
#     st.session_state.history = []
# if "last_uploaded_name" not in st.session_state:
#     st.session_state.last_uploaded_name = None

# # Sidebar
# with st.sidebar:
#     st.header("🧠 Chat Options")
#     if st.session_state.history:
#         st.subheader("📜 Chat History")
#         for i, (q, a) in enumerate(st.session_state.history[::-1]):
#             st.markdown(f"**Q{i+1}:** {q}")
#             st.markdown(f"🟢 {a}")
#         st.markdown("---")
#         chat_data = "\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state.history])
#         st.download_button("📥 Download Chat", chat_data, file_name="chat_history.txt")
    
#     if st.button("🆕 New Chat"):
#         # Reset ChromaDB and session state
#         robust_delete_chroma_db()
#         for key in ["chatbot", "db", "llm", "history", "last_uploaded_name"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.session_state.history = []
#         st.success("✅ Session reset")
#         st.rerun()

# # PDF Upload
# uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# if uploaded_file:
#     if st.session_state.last_uploaded_name != uploaded_file.name:
#         with st.spinner("Processing PDF..."):
#             # Save uploaded file temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 config.DATA_PATH = tmp_file.name

#             try:
#                 # Process PDF
#                 processor = PDFProcessor(config.DATA_PATH)
#                 processed_data = processor.process_pdf()
                
#                 # Save extracted data
#                 output_dir = "./data"
#                 os.makedirs(output_dir, exist_ok=True)
#                 output_path = os.path.join(output_dir, "extracted_data.json")
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     json.dump(processed_data, f, indent=4)

#                 # Load and chunk documents
#                 documents = load_documents()
#                 chunks = split_documents(documents, processed_data=processed_data)
#                 chunks_with_metadata = calculate_chunk_metadata(chunks)

#                 # Initialize ChromaDB
#                 robust_delete_chroma_db()  # Delete any old ChromaDB data
#                 db = add_to_chroma(chunks_with_metadata)

#                 # Initialize LLM
#                 llm = initialize_llm()

#                 # Setup chatbot
#                 st.session_state.db = db
#                 st.session_state.llm = llm
#                 st.session_state.chatbot = ChatHandler(db, llm, config)
#                 st.session_state.history = []
#                 st.session_state.last_uploaded_name = uploaded_file.name

#                 st.success("✅ PDF processed! You can now chat.")
#             except Exception as e:
#                 st.error(f"❌ Error processing PDF: {e}")
#             finally:
#                 if config.DATA_PATH and os.path.exists(config.DATA_PATH):
#                     os.unlink(config.DATA_PATH)

# # Chat Interface
# if "chatbot" in st.session_state:
#     question = st.text_input("Ask a question:")
#     if question:
#         with st.spinner("Generating response..."):
#             try:
#                 response = st.session_state.chatbot.generate_response(question)
#                 st.session_state.chatbot.update_history(question, response)
#                 st.session_state.history.append((question, response))
#             except Exception as e:
#                 st.error(f"❌ Error generating response: {e}")

#     for user_q, bot_a in st.session_state.history:
#         st.markdown(f"**You:** {user_q}")
#         st.markdown(f"**Bot:** {bot_a}")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import json
import tempfile
import shutil
from config import config
from data_processing import PDFProcessor, load_documents, split_documents, calculate_chunk_metadata
from embed_store import add_to_chroma, robust_delete_chroma_db
from initialize import initialize_chromadb, initialize_llm
from chat import ChatHandler

st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄💬 RAG Chatbot for Research Papers")

# Access API keys from Streamlit secrets and set them in config
if "huggingface" in st.secrets and "api_key" in st.secrets["huggingface"]:
    huggingface_api_key = st.secrets["huggingface"]["api_key"]
else:
    huggingface_api_key = None
    st.warning("⚠️ HuggingFace API key not found in secrets")

if "openrouter" in st.secrets and "api_key" in st.secrets["openrouter"]:
    openrouter_api_key = st.secrets["openrouter"]["api_key"]
else:
    openrouter_api_key = None
    st.warning("⚠️ OpenRouter API key not found in secrets")

# Set the API keys in the config
config.set_api_keys(huggingface_api_key=huggingface_api_key, openrouter_api_key=openrouter_api_key)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

# Sidebar
with st.sidebar:
    st.header("🧠 Chat Options")
    if st.session_state.history:
        st.subheader("📜 Chat History")
        for i, (q, a) in enumerate(st.session_state.history[::-1]):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"🟢 {a}")
        st.markdown("---")
        chat_data = "\n".join([f"You: {q}\nBot: {a}" for q, a in st.session_state.history])
        st.download_button("📥 Download Chat", chat_data, file_name="chat_history.txt")
    
    if st.button("🆕 New Chat"):
        # Reset ChromaDB and session state
        robust_delete_chroma_db()
        for key in ["chatbot", "db", "llm", "history", "last_uploaded_name"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.history = []
        st.success("✅ Session reset")
        st.rerun()

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if st.session_state.last_uploaded_name != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                config.DATA_PATH = tmp_file.name

            try:
                # Process PDF
                processor = PDFProcessor(config.DATA_PATH)
                processed_data = processor.process_pdf()
                
                # Save extracted data
                output_dir = "./data"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "extracted_data.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, indent=4)

                # Load and chunk documents
                documents = load_documents()
                chunks = split_documents(documents, processed_data=processed_data)
                chunks_with_metadata = calculate_chunk_metadata(chunks)

                # Initialize ChromaDB
                robust_delete_chroma_db()  # Delete any old ChromaDB data
                db = add_to_chroma(chunks_with_metadata)

                # Initialize LLM
                llm = initialize_llm()

                # Setup chatbot
                st.session_state.db = db
                st.session_state.llm = llm
                st.session_state.chatbot = ChatHandler(db, llm, config)
                st.session_state.history = []
                st.session_state.last_uploaded_name = uploaded_file.name

                st.success("✅ PDF processed! You can now chat.")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")
            finally:
                if config.DATA_PATH and os.path.exists(config.DATA_PATH):
                    os.unlink(config.DATA_PATH)

# Chat Interface
if "chatbot" in st.session_state:
    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.chatbot.generate_response(question)
                st.session_state.chatbot.update_history(question, response)
                st.session_state.history.append((question, response))
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")

    for user_q, bot_a in st.session_state.history:
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**Bot:** {bot_a}")