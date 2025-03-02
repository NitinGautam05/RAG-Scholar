from initialize import initialize_chromadb
from config import config

def add_to_chroma(chunks, db=None):
    """Embed and store chunks in ChromaDB."""
    db = db or initialize_chromadb()
    existing_ids = set(db.get(include=[])["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    else:
        print("No new documents to add")
    return db