import evaluate
import pandas as pd
from typing import List, Dict

# Load the metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Calculation of the BLEU score
def calculate_bleu(reference_text, prediction_text):
    """
    Calculates the BLEU score for a given reference text and prediction.
    :return: BLEU score result as a dictionary
    """
    results = bleu.compute(predictions=prediction_text, references=reference_text)
    return results

# Calculation of the ROUGE score
def calculate_rouge(reference_text, prediction_text):
    """
    Calculates the ROUGE score for a given reference text and a prediction.
    :return: ROUGE score result as a dictionary
    """
    results = rouge.compute(predictions=prediction_text, references=reference_text)
    return results
