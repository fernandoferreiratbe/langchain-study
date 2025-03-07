import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY", default="")

if not openai_key:
    raise ValueError("OpenAI API Key not found")

openai_client = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=openai_key
)

chat_prompt_template = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "Prompt text"),
        ("human", "Prompt text")
    ]
)

messages = chat_prompt_template.format_messages()

print(messages)
