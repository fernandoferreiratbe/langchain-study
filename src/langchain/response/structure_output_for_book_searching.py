"""This agent is used to search for books in the internet according to the user's query."""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class Book(BaseModel):
    title: str = Field(description="The title of the book.")
    author: str = Field(description="The author of the book.")
    price: float = Field(description="The price of the book.")



prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can search for books in the internet."),
    ("user", "Give me the book related to the following query: {query}"),
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Book)

chain = prompt | model

query = "The most sales book, what is the title, author and price?"

response = chain.invoke({"query": query})

logger.info(f"The most sales book is: {response.title} by {response.author} with the price of {response.price}")
