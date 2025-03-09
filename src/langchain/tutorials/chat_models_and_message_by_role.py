import os
from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

load_dotenv()


def validate_whether_openai_api_key_is_defined() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API Key not found")


def get_model_client() -> BaseChatModel:
    model = init_chat_model(model="gpt-4o-mini", model_provider="openai")
    return model


def get_messages(system_prompt: str, human_prompt: str) -> List[BaseMessage]:
    messages = [
        SystemMessage(system_prompt),
        HumanMessage(human_prompt)
    ]
    return messages


def main() -> None:
    validate_whether_openai_api_key_is_defined()

    system_prompt = "You are a specialist in PySpark programming language"
    human_prompt = "How can I read a parquet file using schema definition"

    messages = get_messages(system_prompt=system_prompt, human_prompt=human_prompt)
    chat_model = get_model_client()

    response = chat_model.invoke(messages)

    print(response)


if __name__ == "__main__":
    main()


