# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# # from langchain_core.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# from retrieval import QueryProcessor
# from langchain.docstore.document import Document
# from initialize import initialize_llm
# from config import config

# from langchain.schema import BaseRetriever
# from typing import List
# from pydantic import BaseModel


# # Custom Retriever Class
# class CustomRetriever(BaseRetriever, BaseModel):
#     processor: QueryProcessor  # Explicitly define the processor attribute

#     def get_relevant_documents(self, query: str) -> List[Document]:
#         chunks = self.processor.hybrid_retrieve(query, top_k=config.TOP_K)
#         return [Document(page_content=chunk, metadata=meta) for chunk, meta in chunks]

#     async def aget_relevant_documents(self, query: str) -> List[Document]:
#         # Implement asynchronous retrieval if needed
#         return self.get_relevant_documents(query)


# class ChatRAG:
#     def __init__(self):
#         # Initialize QueryProcessor for retrieval
#         self.processor = QueryProcessor()
#         self.llm = initialize_llm()

#         # Memory for multi-turn conversations
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True,
#             output_key="answer"
#         )

#         # Custom prompt with Chain-of-Thought
#         self.prompt = PromptTemplate(
#             input_variables=["question", "context", "chat_history"],
#             template="""
#             You are an AI assistant answering questions based on research papers.
#             Think step-by-step before responding.

#             **Chat History:**
#             {chat_history}

#             **Query:** {question}

#             **Context from research papers:**
#             {context}

#             **Instructions:**
#             - Answer concisely and accurately.
#             - Use the context and chat history to inform your response.
#             - If the context lacks relevant information, say "The research papers do not contain relevant details."
#             - Do not repeat the context verbatim.

#             **Answer:**
#             """
#         )

#          # Initialize custom retriever
#         self.retriever = CustomRetriever(processor=self.processor)

#         # Define RAG chain with custom retriever
#         self.chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             # retriever=custom_retriever,
#             retriever=self.retriever,
#             memory=self.memory,
#             combine_docs_chain_kwargs={"prompt": self.prompt},
#             return_source_documents=True,
#             output_key="answer"
#         )

#     def ask(self, query):
#         """Process a query and return the answer with sources."""
#         try:
#             result = self.chain({"question": query})
#             return {
#                 "answer": result["answer"],
#                 "sources": [doc.page_content for doc in result["source_documents"]]
#             }
#         except Exception as e:
#             return {"answer": f"Error: {str(e)}", "sources": []}


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from retrieval import QueryProcessor
from langchain.docstore.document import Document
from initialize import initialize_llm
from config import config

from langchain.schema import BaseRetriever
from typing import List
from pydantic import BaseModel


class CustomRetriever(BaseRetriever, BaseModel):
    processor: QueryProcessor

    def get_relevant_documents(self, query: str) -> List[Document]:
        chunks = self.processor.hybrid_retrieve(query, top_k=config.TOP_K)
        return [Document(page_content=chunk, metadata=meta) for chunk, meta in chunks]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


class ChatRAG:
    def __init__(self):
        self.processor = QueryProcessor()
        self.llm = initialize_llm()
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Improved prompt for better responses
        self.prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""
            You are a precise and knowledgeable AI assistant answering questions based on research papers.
            
            **Chat History:**
            {chat_history}

            **User Query:** {question}

            **Context from research papers:**
            {context}

            **Instructions:**
            1. Analyze the query and context thoroughly.
            2. Focus on information directly related to the query.
            3. If the papers contain tables, cite the table data specifically.
            4. If the papers contain formulas, explain them clearly.
            5. If the context doesn't contain relevant information, state "The research papers do not contain relevant information about this specific query."
            6. Provide a structured, concise answer with clear sections if appropriate.
            7. Avoid general knowledge not found in the context.
            8. Include specific page references or section information when available.

            **Answer:**
            """
        )

        self.retriever = CustomRetriever(processor=self.processor)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            output_key="answer"
        )

    def ask(self, query):
        """Process a query and return the answer with sources."""
        try:
            # Preprocess query if enabled in config
            if hasattr(config, 'QUERY_REWRITE') and config.QUERY_REWRITE:
                query = self.processor.rewrite_query(query)
                
            result = self.chain({"question": query})
            
            # Format sources for better readability
            sources = []
            for doc in result["source_documents"]:
                source_type = doc.metadata.get('source', 'unknown')
                source_info = f"[{source_type}] {doc.page_content[:100]}..."
                sources.append(source_info)
                
            return {
                "answer": result["answer"],
                "sources": sources
            }
        except Exception as e:
            return {"answer": f"Error processing your query: {str(e)}", "sources": []}