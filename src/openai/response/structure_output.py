import json
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions and help with tasks."
        },
        {
            "role": "user",
            "content": "What is the capital of the moon?"
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "MoonCapitalInfo",
            "description": "Information about the moon's capital",
            "schema": {
                "type": "object",
                "properties": {
                    "capital": {
                        "type": "string",
                        "description": "The capital city of the moon"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation about why this is or isn't a capital"
                    }
                },
                "required": ["capital", "explanation"]
            }
        }
    }
)

print(json.loads(response.choices[0].message.content)["capital"])