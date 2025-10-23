import GPTModelsAPI
import OllamaClient
import OllamaModelsLocal

def model_wrapper(prompt: str, model: str):
    """
    Führt den passenden Modellaufruf aus, abhängig von Modell und ollamaLocal-Flag.

    :param prompt: Der Eingabetext / Prompt
    :param model: Modellname (z.B. 'gpt-4', 'gpt-5', 'qwen_4b', 'gemma_7b', ...)
    :return: Antwort des Modells als String
    """

    ollamaLocal = False  # Globale Umschaltung für lokale vs. externe Ollama-Modelle

    # 1OpenAI GPT-Modelle (enthält 'gpt-4' oder 'gpt-5')
    if "gpt-4" in model.lower() or "gpt-5" in model.lower():
        return GPTModelsAPI.run_gpt_openai(prompt, model)

    # Ollama-Client (remote), wenn ollamaLocal = False
    elif not ollamaLocal:
        return OllamaClient.ollama_client(prompt, model)

    # 3Lokale Ollama-Modelle, wenn ollamaLocal = True
    else:
        return OllamaModelsLocal.run_Ollama_Model(prompt, model)












    #output = mo.run_gemma_7b(p.build_prompt(text_segment['original_text']))    
    #output = mo.run_qwen_4b(p.build_prompt(text_segment['original_text']))    
    #output = mo.run_llama3_2_1b(p.build_prompt(text_segment['original_text']))
    #output = mo.run_llama3_2_3b(p.build_prompt(text_segment['original_text']))
    #output = mo.run_llama2_7b(p.build_prompt(text_segment['original_text']))
    #output = mo.run_gemma3_1b(p.build_prompt(text_segment['original_text']))
    #output = mo.run_gemma3_4b(p.build_prompt(text_segment['original_text']))
    #output = mo.run_deepseek_r1_1_5_b(p.build_prompt(text_segment['original_text']))
    #output = mo.run_gpt_openai(p.build_prompt(text_segment['original_text']))