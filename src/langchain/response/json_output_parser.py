import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    """" Client for Open AI """

    def __init__(self):
        """ OPENAI_API_KEY has been previously defined and ChatOpenAI knows how to get it  """
        self.chat_open_ai = ChatOpenAI(
            model="gpt-4o",
            temperature=0.5
        )

    @staticmethod
    def _build_prompt_template() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert software development"),
                ("human", "Write a snippet to put a message into a sqs "
                          "queue using a uuid for MessageGroupId using language {language}")
            ]
        )

    def get_snippet_for(self, programming_language: str):
        if not programming_language:
            raise ValueError(f"Programming Language '{programming_language}' informed is invalid.")

        llm = self._build_prompt_template() | self.chat_open_ai

        response = llm.invoke({"language": programming_language})

        return response


if __name__ == "__main__":
    client = OpenAIClient()
    snippet = client.get_snippet_for(programming_language="python")
    print(snippet)
