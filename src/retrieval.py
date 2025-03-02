# from initialize import initialize_chromadb, initialize_llm
# from langchain_huggingface import HuggingFaceEmbeddings
# from config import config
# from rank_bm25 import BM25Okapi
# from langchain.prompts import PromptTemplate
# from langchain.chains.summarize import load_summarize_chain
# from langchain.docstore.document import Document

# class QueryProcessor:
#     def __init__(self):
#         self.vector_store = initialize_chromadb()
#         self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
#         self.llm = initialize_llm()
#         retrieved_data = self.vector_store.get()
#         self.documents = retrieved_data["documents"]
#         self.metadata = retrieved_data["metadatas"]
#         self.tokenized_docs = [doc.split(" ") for doc in self.documents]
#         self.bm25 = BM25Okapi(self.tokenized_docs)

#     def embed_query(self, query):
#         return self.embeddings.embed_query(query)

#     def hybrid_retrieve(self, query, top_k=config.TOP_K):
#         # BM25
#         bm25_scores = self.bm25.get_scores(query.split(" "))
#         bm25_top = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
#         bm25_results = [(self.documents[i], self.metadata[i], s) for i, s in bm25_top]

#         # Vector
#         query_embedding = self.embed_query(query)
#         vector_results = self.vector_store.similarity_search_by_vector(query_embedding, k=top_k)
#         # print(vector_results)  # Inspect the structure of the results
        
#         # Handle cases where Document objects don't have a 'score' attribute
#         vector_chunks = []
#         for doc in vector_results:
#             score = getattr(doc, "score", 1.0)  # Default score of 1.0 if 'score' is missing
#             vector_chunks.append((doc.page_content, doc.metadata, score))

#         # Combine and re-rank
#         combined = {}
#         for chunk, meta, score in bm25_results + vector_chunks:
#             if chunk not in combined:
#                 combined[chunk] = (meta, score)
#             else:
#                 combined[chunk] = (meta, max(combined[chunk][1], score))

#         ranked = sorted(combined.items(), key=lambda x: x[1][1], reverse=True)[:top_k]
#         return [(chunk, meta) for chunk, (meta, _) in ranked]

#     def summarize_chunks(self, chunks):
#         docs = [Document(page_content=chunk) for chunk, _ in chunks]
#         chain = load_summarize_chain(self.llm, chain_type="map_reduce")
#         return chain.run(docs)

#     def generate_response(self, query):
#         relevant_chunks = self.hybrid_retrieve(query)
#         if not relevant_chunks:
#             return "I couldn't find relevant information in the research papers."
#         summary = self.summarize_chunks(relevant_chunks[:3])
#         context = "\n".join([f"[{m['source']}] {c}" for c, m in relevant_chunks[:3]])
#         prompt = PromptTemplate(
#             input_variables=["query", "context", "summary"],
#             template="""
#             You are an AI assistant answering questions based on research papers.
#             Think step-by-step before answering.

#             **Query:** {query}
#             **Context from research papers:** {context}
#             **Summary:** {summary}
#             **Instructions:**
#             - Answer concisely and accurately.
#             - Use the summary to guide your response.
#             - If the context lacks relevant information, respond with "The research papers do not contain relevant details."
#             - Do not repeat the context verbatim.
#             **Answer:** 
#             """
#         )
#         response = self.llm.invoke(prompt.format(query=query, context=context, summary=summary))
#         return response.content if hasattr(response, "content") else str(response)

# if __name__ == "__main__":
#     processor = QueryProcessor()
#     user_query = "Explain the methodology of the research paper."
#     response = processor.generate_response(user_query)
#     print("\n📝 Response:\n", response)



from initialize import initialize_chromadb, initialize_llm
from langchain_huggingface import HuggingFaceEmbeddings
from config import config
from rank_bm25 import BM25Okapi
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        self.vector_store = initialize_chromadb()
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.llm = initialize_llm()
        
        # Load and prepare documents for BM25
        try:
            retrieved_data = self.vector_store.get()
            self.documents = retrieved_data["documents"]
            self.metadata = retrieved_data["metadatas"]
            # print(f"Stored documents: {len(self.documents)}")
            # print(f"Example document: {self.documents[0] if self.documents else 'No documents found'}")

            
            # Tokenize documents for BM25
            self.tokenized_docs = [doc.split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"Initialized with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error initializing documents: {e}")
            # Initialize with empty collections if no documents are found
            self.documents = []
            self.metadata = []
            self.tokenized_docs = []
            self.bm25 = None
            logger.warning("Initialized with empty document set")

    def rewrite_query(self, query):
        """Rewrite the query to make it more suitable for retrieval."""
        if not hasattr(config, 'QUERY_REWRITE') or not config.QUERY_REWRITE:
            return query
            
        # Simple query rewriting
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You need to rewrite the following query to make it more effective for retrieval from research papers.
            Original Query: {query}
            
            Think about key terms and concepts that might appear in the relevant sections.
            Rewritten Query:
            """
        )
        
        try:
            response = self.llm.invoke(prompt.format(query=query))
            rewritten = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query  # Fall back to original query

    def embed_query(self, query):
        """Embed the query for vector search."""
        return self.embeddings.embed_query(query)

    def hybrid_retrieve(self, query, top_k=config.TOP_K):
        """Enhanced hybrid retrieval with proper weighting."""
        # Check if we have documents to search
        if not self.documents:
            logger.warning("No documents available for retrieval")
            return []
            
        # Get vector search results
        query_embedding = self.embed_query(query)
        vector_results = self.vector_store.similarity_search_by_vector(
            query_embedding, 
            k=top_k*2  # Retrieve more and then re-rank
        )
        
        # Process vector results
        vector_chunks = []
        for doc in vector_results:
            score = getattr(doc, "score", None)
            if score is None:
                # If no explicit score, use a default high score
                score = 0.8
            vector_chunks.append((doc.page_content, doc.metadata, score))
            
        # Get BM25 results if available
        bm25_results = []
        if self.bm25:
            try:
                query_tokens = query.split(" ")
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Normalize BM25 scores to 0-1 range
                if len(bm25_scores) > 0:
                    max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
                    bm25_scores = [s/max_score for s in bm25_scores]
                
                # Get top results
                bm25_top = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:top_k*2]
                bm25_results = [(self.documents[i], self.metadata[i], s) for i, s in bm25_top if s > 0.1]
            except Exception as e:
                logger.error(f"BM25 retrieval error: {e}")
                
        # Weight configuration for hybrid search
        bm25_weight = getattr(config, 'BM25_WEIGHT', 0.4)
        vector_weight = getattr(config, 'VECTOR_WEIGHT', 0.6)
        
        # Combine and re-rank with weighted scores
        combined = {}
        
        # Add vector results with their weight
        for chunk, meta, score in vector_chunks:
            combined[chunk] = {
                'meta': meta,
                'score': vector_weight * score
            }
            
        # Add BM25 results, combining scores when the same chunk appears in both
        for chunk, meta, score in bm25_results:
            if chunk in combined:
                # Add the weighted BM25 score
                combined[chunk]['score'] += bm25_weight * score
            else:
                combined[chunk] = {
                    'meta': meta,
                    'score': bm25_weight * score
                }
                
        # Rank the results by final score
        ranked = sorted(combined.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
        
        # Extract just the chunks and metadata
        result = [(chunk, data['meta']) for chunk, data in ranked]
        
        logger.info(f"Retrieved {len(result)} chunks for query: {query[:50]}...")
        return result

    def summarize_chunks(self, chunks):
        """Summarize retrieved chunks."""
        if not chunks:
            return "No relevant information found."
            
        docs = [Document(page_content=chunk) for chunk, _ in chunks]
        try:
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            return chain.run(docs)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple concatenation if summarization fails
            return "\n\n".join([doc.page_content for doc in docs[:2]])

    def generate_response(self, query):
        """Generate a response to the query based on retrieved documents."""
        # Get relevant chunks
        relevant_chunks = self.hybrid_retrieve(query)
        if not relevant_chunks:
            logger.warning("No relevant chunks found for the query.")

        
        if not relevant_chunks:
            return "I couldn't find relevant information in the research papers."
            
        # Format context with source information
        context_parts = []  # Prepare context parts for response

        for i, (chunk, meta) in enumerate(relevant_chunks[:3]):
            source_type = meta.get('source', 'text')
            page = meta.get('page', 'unknown page')
            
            if source_type == 'table':
                context_parts.append(f"[Table from page {page}]\n{chunk}")
            elif source_type == 'formula':
                context_parts.append(f"[Formula from page {page}]\n{chunk}")
            else:
                context_parts.append(f"[Content from page {page}]\n{chunk}")
                
        context = "\n\n".join(context_parts)
        
        # Generate a concise summary 
        summary = self.summarize_chunks(relevant_chunks[:3])
        
        # Improved prompt template
        prompt = PromptTemplate(  # Create a prompt template for response generation

            input_variables=["query", "context", "summary"],
            template="""
            You are a precise AI assistant answering questions about research papers.
            
            **Query:** {query}
            
            **Context from research papers:** 
            {context}
            
            **Summary of key information:** 
            {summary}
            
            **Instructions:**
            - Ensure to provide citations for the sources used in your answer.
            1. Focus on directly answering the query using ONLY information from the provided context.
            2. Structure your response clearly with headings if appropriate.
            3. If discussing tables, present the data in a well-organized manner.
            4. If discussing formulas, explain their meaning and components clearly.
            5. If the context doesn't address the query, state "The research papers don't contain relevant information about this query."
            6. Cite the page numbers in your answer, e.g. "(page 3)"
            7. Be specific and precise rather than general. """
        )
        response = self.llm.invoke(prompt.format(query=query, context=context, summary=summary))
        return response.content if hasattr(response, "content") else str(response)

if __name__ == "__main__":
    processor = QueryProcessor()
    user_query = "Explain the methodology of the research paper."
    response = processor.generate_response(user_query)
    print("\n📝 Response:\n", response)
