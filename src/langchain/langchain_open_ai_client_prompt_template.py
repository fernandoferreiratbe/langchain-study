import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


print(OPENAI_API_KEY)

exit

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

modelo_de_prompt = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {dias} dias, para uma família com {criancas} crianças, que gostam de {atividade}."
)

prompt = modelo_de_prompt.format(
    dias=numero_de_dias,
    criancas=numero_de_criancas,
    atividade=atividade
)

print(prompt)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=OPENAI_API_KEY
)

resposta = llm.invoke(prompt)

print(resposta.content)
