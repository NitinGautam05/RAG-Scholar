# RAG-Based Chatbot for Research Papers

## 📌 Project Overview
This project is a **Retrieval-Augmented Generation (RAG) chatbot** designed to process and answer queries from **research papers (PDFs)**. It extracts relevant information from scientific documents containing **text, tables, images, and formulas** to provide meaningful responses.

## 🚀 Features
- **PDF Processing**: Extracts and preprocesses content from research papers.
- **Embeddings & Vector Search**: Uses **sentence-transformers/all-MiniLM-L6-v2** for embedding generation and **ChromaDB** for efficient vector storage and retrieval.
- **LLM Integration**: Powered by **TinyLlama-1.1B-Chat-v1.0** via **Hugging Face API** for contextual answers.
- **Efficient Query Handling**: Retrieves the most relevant document chunks before passing them to the LLM.
- **Scalable & Modular**: Clean and structured code for easy modifications and improvements.

## 🛠️ Tech Stack
- **Python** (Primary Language)
- **LangChain** (Framework for RAG)
- **ChromaDB** (Vector Database)
- **Hugging Face API** (LLM & Embeddings)
- **PyMuPDF** (PDF Processing)
- **Streamlit** (For deployment)

## 📂 Project Structure
```
📁 rag-chatbot
│── 📂 data                # Stores processed PDF data
│── 📂 DB                  # ChromaDB vector storage
│── 📂 src                 # Source code
│── requirements.txt       # Dependencies
│── README.md              # Project Documentation
```

