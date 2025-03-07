""" Agent sample define a tool that search for contents at web """
import os
from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv


load_dotenv()


def search_at_google(query: str) -> Any:
    return f"Response for {query}"


# Tool definition
tools = [
    Tool(name="Google", func=search_at_google, description="Search for information at web")
]

# Agent initialization
openai_api_key = os.getenv("OPENAI_API_KEY", default="")
if not openai_api_key:
    raise ValueError("API Key not found")

chat = ChatOpenAI(openai_api_key=openai_api_key)
agent = initialize_agent(
    tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Run Agent
response = agent.run("Who discovery Brazil?")

print(response)

