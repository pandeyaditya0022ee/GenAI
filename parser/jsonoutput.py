# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()
#open source model usage example
# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm = llm)

#gemini model usage example
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

parser = JsonOutputParser()
template = PromptTemplate(
    template="Write a short story about {topic} in the style of {style}. \n {format_instructions} ",
    input_variables=["topic", "style"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    
)

chain = template | model | parser
result = chain.invoke({'topic': "a brave knight", 'style': "Shakespearean"})

print("Result:", result)