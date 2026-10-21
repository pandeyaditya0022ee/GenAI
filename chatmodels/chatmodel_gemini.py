from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.2,
    max_output_tokens=1000,
)

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = model.invoke(user_input)

    print("AI:", response.text)