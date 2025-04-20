# from initialize import initialize_chromadb
# from config import config
# import shutil
# import os
# import gc
# import streamlit as st
# import time


# def add_to_chroma(chunks, db=None):
#     db = db or initialize_chromadb()
#     existing_ids = set(db.get(include=[])["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
#     if new_chunks:
#         print(f"Adding new documents: {len(new_chunks)}")
#         db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
#     else:
#         print("No new documents to add")
#     return db

# def robust_delete_chroma_db():
#     db_path = "D:/rag-chatbot/DB/chromadb"
#     data_paths = [
#         "D:/rag-chatbot/data/output",
#         "D:/rag-chatbot/data/tables",
#         "D:/rag-chatbot/data/pages"
#     ]
    
#     max_retries = 5
#     retry_delay = 1  # seconds

#     for _ in range(max_retries):
#         all_deleted = True

#         if os.path.exists(db_path):
#             shutil.rmtree(db_path, ignore_errors=True)
#             all_deleted &= not os.path.exists(db_path)

#         for path in data_paths:
#             if os.path.exists(path):
#                 shutil.rmtree(path, ignore_errors=True)
#                 all_deleted &= not os.path.exists(path)

#         if all_deleted:
#             return True
        
#         time.sleep(retry_delay)
    
#     return False

from initialize import initialize_chromadb
from config import config
import shutil
import os
import time

# def add_to_chroma(chunks, db=None):
#     db = db or initialize_chromadb()
#     existing_ids = set(db.get(include=[])["ids"])
#     new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
#     if new_chunks:
#         db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
#     return db

def add_to_chroma(chunks, db=None):
    print("[DEBUG] Inside add_to_chroma")
    db = db or initialize_chromadb()
    
    try:
        print("[DEBUG] Attempting to get existing documents")
        existing_ids = set(db.get(include=[])["ids"])  # ❗ This line triggers "no such table"
    except Exception as e:
        print(f"[ERROR] ChromaDB get() failed: {e}")
        raise

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"[DEBUG] Adding {len(new_chunks)} new chunks")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    else:
        print("[DEBUG] No new chunks to add")

    return db


def robust_delete_chroma_db():
    db_path = config.CHROMADB_PATH
    data_paths = [
        "./data/output",
        "./data/tables",
        "./data/pages"
    ]
    max_retries = 5
    retry_delay = 1
    for _ in range(max_retries):
        all_deleted = True
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
            all_deleted &= not os.path.exists(db_path)
        for path in data_paths:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                all_deleted &= not os.path.exists(path)
        if all_deleted:
            return True
        time.sleep(retry_delay)
    return False