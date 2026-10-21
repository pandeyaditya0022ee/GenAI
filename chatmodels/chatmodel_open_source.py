#Chat Models

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

chat_history = []
while(True):
    query = input("YOU: ")
    chat_history.append(("user", query))
    if query.lower() == "exit":
        break
    result = model.invoke(chat_history)
    print(f"ASSISTANT: {result.content}")
    chat_history.append(("assistant", result.content))
    print("--------------------------------------------------")
    
    
#Embeddings Models
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# texts = ["This is a test sentence.", "This is another test sentence."]
# vector = embeddings.embed_documents(texts)

# print(vector)