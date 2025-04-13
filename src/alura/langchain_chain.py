import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

load_dotenv()
# set_debug helps us to see what's going on through the looping of the prompts execution
set_debug(True)


llm = ChatOpenAI(
    # model="gpt-4o", Its response has a lot of cities around the world. To reproduce the exercise use 3.5 turbo
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)


city_model = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interest}"
)

restaurant_model = ChatPromptTemplate.from_template(
    "Sugira restaurantes popoulares entre locais em {city}"
)

cultural_model = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {city}"
)

city_chain = LLMChain(prompt=city_model, llm=llm)
restaurant_chain = LLMChain(prompt=restaurant_model, llm=llm)
cultural_chain = LLMChain(prompt=cultural_model, llm=llm)

simple_sequential_chain = SimpleSequentialChain(chains=[city_chain, restaurant_chain, cultural_chain], verbose=True)

response = simple_sequential_chain.invoke({"input": "prais"})

print(response)
