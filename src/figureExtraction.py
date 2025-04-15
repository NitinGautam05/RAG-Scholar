# from llama_cpp import Llama

# llm = Llama(
#     model_path="D:\\rag-chatbot\\model\\mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     n_threads=8,
#     n_ctx=512,
#     n_gpu_layers=0,
#     verbose=True
# )

# output = llm("### Instruction:\nWhat is the capital of Germany?\n\n### Response:", max_tokens=50)
# print(output["choices"][0]["text"].strip())

from llama_cpp import Llama

llm = Llama(model_path="D:\\rag-chatbot\\model\\mistral-7b-instruct-v0.1.Q4_K_M.gguf")
print("Max context length:", llm.context_params.n_ctx)
