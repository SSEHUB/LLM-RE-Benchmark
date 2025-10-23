import os
import GPTModelsAPI
import OllamaModelsLocal
import OllamaClient
from openai import OpenAI

client = OpenAI()


def run_gpt_openai(prompt: str, gptmodel: str) -> str:
    """
    Sends a prompt to the OpenAI GPT model and returns the response.
    :param prompt: The input text to be processed by the model
    :return: The generated response from the model (always a string)
    """
    try:
        response = client.chat.completions.create(
            model=gptmodel,  # "gpt-4", je nach gewünschtem Modell
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error querying OpenAI GPT model: {e}")
        return ""
