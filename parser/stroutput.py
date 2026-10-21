from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()
#open source model usage example
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

#gemini model usage example
# model = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash"
# )
template1 = PromptTemplate(
    template="Write a short story about {topic} in the style of {style}.",
    input_variables=["topic", "style"]
)
template2 = PromptTemplate(
    template="Lets Summarize {text}.",
    input_variables=["text"]
)
#Standard usage example
# prompt1 = template1.invoke({'topic': "a brave knight", 'style': "Shakespearean"})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text': result1.content})

# result2 = model.invoke(prompt2)


# print("Result 2:", result2.content)

#Using StructuredOutputParser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "a brave knight", 'style': "Shakespearean"})

print("Result:", result)