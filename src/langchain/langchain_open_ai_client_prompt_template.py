import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", default="")

if not OPENAI_API_KEY:
    raise ValueError("Open AI Key not found")

days_number = 7
child_number = 2
activity = "praia"

prompt_template = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {days} days, para uma família com {child} crianças, que gostam de {activity}."
)

prompt = prompt_template.format(
    days=days_number,
    child=child_number,
    activity=activity
)

print(prompt)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=OPENAI_API_KEY
)

response = llm.invoke(prompt)

print(response.content)
