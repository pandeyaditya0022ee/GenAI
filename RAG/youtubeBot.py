from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from urllib.parse import urlparse, parse_qs
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

url = input("Paste video url here ->  ")

parsed_url = urlparse(url)

video_id = parse_qs(parsed_url.query)["v"][0]


try:
    transcript_list = (
        YouTubeTranscriptApi().fetch(video_id, languages=["en"]).to_raw_data()
    )

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except Exception as e:
    print(type(e).__name__)
    print(e)
    exit()


# Splitting transcript into chunks so that we store it into vector store

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Embedding Generation and storing in vector store
class OpenRouterEmbeddings(Embeddings):

    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/embeddings"

    def _embed(self, texts):

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            },
        )

        response.raise_for_status()

        data = response.json()

        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]
    
embeddings = OpenRouterEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="nvidia/nemotron-3-embed-1b:free"
)
vector_store = FAISS.from_documents(chunks, embeddings)


# Retrival

retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Augementation

llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.
      {context}
      Question : {question}
    """,
    input_variables=["context", "question"],
)

# Forming chains


def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


# creating a parallel chain
parallel_chain = RunnableParallel(
    {
        "context": retriver | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser



while(True):
    question = input("YOU : ")
    
    if(question == "exit") :
        break
    
    result = main_chain.invoke(question)
    
    print(result)
