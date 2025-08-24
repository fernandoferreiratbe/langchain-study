import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

anthropic_model = "claude-3-5-sonnet-20240620"
llm = ChatAnthropic(
    model=anthropic_model,
    temperature=0.7
)

messages = [
    SystemMessage("Você é um assistente prestativo. Responda em Português."),
    HumanMessage("Qual o seu nome?")
]

response = llm.invoke(messages)
print(response.content)

print("-------------------------------------------------------------------")

prompt_chat_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente prestativo. Responda em Português."),
    ("human", "{question}")
])

chain = prompt_chat_template | llm

response_chain = chain.invoke({
    "question": "Quem é o presidente do Brasil?"
})

print(response_chain.content)