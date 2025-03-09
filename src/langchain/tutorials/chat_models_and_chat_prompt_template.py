import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel


def verify_whether_openai_api_key_is_defined() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Open AI API Key not found.")


def create_openai_chat() -> BaseChatModel:
    verify_whether_openai_api_key_is_defined()
    return init_chat_model(model="gpt-4o-mini", model_provider="openai")


def create_chat_prompt_template() -> ChatPromptTemplate:
    system_template = "Translate the following from English into {language}"

    return ChatPromptTemplate(
        [
            ("system", system_template),
            ("user", "{text}")
        ]
    )


def main() -> None:
    prompt_template = create_chat_prompt_template()
    prompt = prompt_template.invoke({"language": "Portuguese", "text": "hi!"})

    print(prompt)

    chat = create_openai_chat()
    response = chat.invoke(prompt)

    print(response)


if __name__ == "__main__":
    main()
