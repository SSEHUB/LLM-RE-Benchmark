import GPTModelsAPI
import OllamaClient
import OllamaModelsLocal

def model_wrapper(prompt: str, model: str):
    """
    Executes the appropriate model call, depending on the model and ollamaLocal flag.

    :param prompt: The input text / prompt
    :param model: Model name (e.g., ‘gpt-4’, ‘gpt-5’, ‘qwen_4b’, ‘gemma_7b’, ...)
    :return: Model response as a string
    """

    ollamaLocal = False  # Global switch for local vs. external Ollama models

    # 1OpenAI GPT-models (contains 'gpt-4' or 'gpt-5')
    if "gpt-4" in model.lower() or "gpt-5" in model.lower():
        return GPTModelsAPI.run_gpt_openai(prompt, model)

    # Ollama-Client (remote), if ollamaLocal = False
    elif not ollamaLocal:
        return OllamaClient.ollama_client(prompt, model)

    # 3 local Ollama-Models, if ollamaLocal = True
    else:
        return OllamaModelsLocal.run_Ollama_Model(prompt, model)
