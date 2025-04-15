# class ChatHandler:
#     def __init__(self, db, llm, config):
#         self.db = db
#         self.llm = llm
#         self.config = config
#         self.conversation_history = []
#         self.MAX_HISTORY = 5 

#     def format_conversation_history(self):
#         """Format the conversation history into a string for the prompt."""
#         if not self.conversation_history:
#             return ""
#         formatted = "\n\n--- Previous Conversation ---\n"
#         for human_msg, ai_msg in self.conversation_history:
#             formatted += f"Human: {human_msg}\nAI: {ai_msg}\n"
#         formatted += "--- End of Previous Conversation ---\n"
#         return formatted

#     def generate_response(self, query):
#         """Generate a response for the given query using the RAG pipeline."""
#         # Perform similarity search to retrieve relevant documents
#         docs = self.db.similarity_search(query, k=self.config.TOP_K)
#         unique_docs = list({doc.page_content: doc for doc in docs}.values())
#         context = "\n\n".join(doc.page_content for doc in unique_docs[:self.config.TOP_K])

#         # Format the conversation history
#         history_str = self.format_conversation_history()

#         # Construct the prompt with history and current query
#         prompt = (
#         "<|user|>\n"
#         "You are a conversational assistant helping with questions about a research paper. "
#         "Use only the provided context to answer questions accurately and concisely. "
#         "Maintain a natural, conversational tone and refer to previous exchanges if relevant. "
#         "If the context is unclear or insufficient, say so and don’t guess.\n"
#         f"{history_str}"
#         f"Context from the paper:\n{context}\n"
#         f"Current Question: {query}\n"
#         "<|assistant|>\n"
#     )

#         # Get the LLM response
#         response = self.llm.invoke(prompt)
#         return response.content.strip()

#     def update_history(self, query, response):
#         """Update the conversation history with the latest query and response."""
#         self.conversation_history.append((query, response))
#         # Keep only the last MAX_HISTORY exchanges to avoid overly long prompts
#         if len(self.conversation_history) > self.MAX_HISTORY:
#             self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]

#     def run(self):
#         """Run the conversational loop."""
#         print("Hey there! I'm ready to chat about the research paper. Ask me anything! (Type 'quit' to exit)")

#         while True:
#             query = input("\nYou: ")
#             if query.lower() == "quit":
#                 print("Thanks for chatting! Catch you later!")
#                 break

#             # Generate response and update history
#             response = self.generate_response(query)
#             print("\nBot:", response)
#             self.update_history(query, response)

class ChatHandler:
    def __init__(self, db, llm, config):
        self.db = db
        self.llm = llm
        self.config = config
        self.conversation_history = []
        self.MAX_HISTORY = 5

    def format_conversation_history(self):
        """Format the conversation history."""
        if not self.conversation_history:
            return ""
        formatted = "\n\n--- Previous Conversation ---\n"
        for human_msg, ai_msg in self.conversation_history:
            formatted += f"Human: {human_msg}\nAI: {ai_msg}\n"
        formatted += "--- End of Previous Conversation ---\n"
        return formatted

    def generate_response(self, query):
        """Generate a response using the RAG pipeline."""
        docs = self.db.similarity_search(query, k=self.config.TOP_K)
        unique_docs = list({doc.page_content: doc for doc in docs}.values())
        context = "\n\n".join(doc.page_content for doc in unique_docs[:self.config.TOP_K])
        history_str = self.format_conversation_history()
        prompt = (
            "<|user|>\n"
            "You are a conversational assistant for research papers. "
            "Answer using only the provided context, accurately and concisely. "
            "Use a natural tone and refer to previous exchanges if relevant. "
            "If the context is unclear, say so and avoid guessing.\n"
            f"{history_str}"
            f"Context:\n{context}\n"
            f"Question: {query}\n"
            "<|assistant|>\n"
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def update_history(self, query, response):
        """Update conversation history."""
        self.conversation_history.append((query, response))
        if len(self.conversation_history) > self.MAX_HISTORY:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]