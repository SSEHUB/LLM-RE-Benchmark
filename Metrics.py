import evaluate
import pandas as pd
from typing import List, Dict

# Lade die Metriken
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Berechnung des BLEU-Scores
def calculate_bleu(reference_text, prediction_text):
    """
    Berechnet den BLEU-Score für eine gegebene Referenztext und eine Vorhersage.
    :return: BLEU-Score-Ergebnis als Dictionary
    """
    results = bleu.compute(predictions=prediction_text, references=reference_text)
    return results

# Berechnung des ROUGE-Scores
def calculate_rouge(reference_text, prediction_text):
    """
    Berechnet den ROUGE-Score für eine gegebene Referenztext und eine Vorhersage.
    :return: ROUGE-Score-Ergebnis als Dictionary
    """
    results = rouge.compute(predictions=prediction_text, references=reference_text)
    return results
