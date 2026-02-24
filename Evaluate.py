import Prompts as p
import DataLoader as dl
import Metrics as me
import MonitorResults as mr
import pandas as pd
import ModelWrapper as mw
import os
import re

from typing import List, Dict, Set

results_per_textsegment_of_all_runs = {}



def evaluate_dataset(json_file: str, run_id: int, metric_score_threshold: float, model: str):

    data = dl.load_json_to_dict(json_file)

    if not data:
        print("No data found in JSON file.")
        return

    print(f"\n✅ Loaded {len(data)} text segments. Starting evaluation...\n")

    for text_id, entry in data.items():
        print(f"🔹 Evaluating text_id: {text_id}")
        evaluate_text_segment(text_id, entry, run_id, metric_score_threshold, model)
        print("-" * 50)
        #break



def evaluate_text_segment(text_id: str, text_segment: dict, run_id: int, metric_score_threshold: float, model: str):


    output = mw.model_wrapper(p.build_prompt(text_segment['original_text']), model)
    #output = mo.run_gpt_openai(p.build_prompt(text_segment['original_text']))

    # Split the llm output string into a list of requirements
    list_of_predictions = split_llm_output_numbered(output)

    # Berechnung aller Metrikwerte (Rouge) auf einmal --> Output: Matrix zwischen allen Predictions und Referenzen (wenn keine Referenz vorhanden, wird der Hinweissatz "This text does not ..." als Referenz eingeführt)
    similarity_scores = calculate_all_metrics_for_text_segment(list_of_predictions, extract_reference_with_placeholder(text_segment["requirements"]))

    # Berechnung der Results für eine Textsegment
    text_segment_results = determine_text_segment_matchings(list_of_predictions, extract_reference_with_placeholder(text_segment["requirements"]), similarity_scores, metric_score_threshold)

    # Ergebnisse speichern
    results_per_textsegment_of_all_runs[len(results_per_textsegment_of_all_runs)] = {
        "run_id": run_id,
        "text_id": text_id,
        "model": model,
        "metric_score_threshold": metric_score_threshold,
        "total_references": count_valid_references(extract_reference_with_placeholder(text_segment["requirements"])), # Placerholder ("This text does not ...") must be removed when counting total references. 
        "unmatched_references": text_segment_results["unmatched_references"],
        "total_predictions": text_segment_results["total_predictions"], # Placerholder ("This text does not ...") must be removed when counting total predictions.
        "unmatched_predictions": text_segment_results["unmatched_predictions"],
        "output": output
    }



def count_valid_predictions(predictions: List[str]) -> int:
    """
    Gibt 0 zurück, wenn die Liste nur eine Vorhersage enthält und diese leer ist,
    oder den speziellen Hinweistext enthält. Ansonsten wird die Anzahl der Vorhersagen zurückgegeben.
    """
    placeholder = "This text does not contain any functional requirement information"

    if len(predictions) == 1:
        cleaned = predictions[0].strip()
        if cleaned == "" or cleaned == placeholder:
            return 0

    return len(predictions)



def count_valid_references(reference_groups: List[List[str]]) -> int:
    """
    Gibt 0 zurück, wenn nur eine Referenzgruppe vorhanden ist und diese leer ist,
    oder nur den Hinweistext enthält. Ansonsten wird die Anzahl der Referenzgruppen zurückgegeben.
    """
    placeholder = "This text does not contain any functional requirement information."

    if len(reference_groups) == 1:
        single_ref = reference_groups[0]
        # Bereinige leere Einträge ("" oder None)
        cleaned = [r.strip() for r in single_ref if r and r.strip()]
        if not cleaned or (len(cleaned) == 1 and cleaned[0] == placeholder):
            return 0

    return len(reference_groups)



def determine_text_segment_matchings(predictions: List[str], references: List[List[str]], similarity_scores: dict, metric_score_threshold: float) -> dict:
    """
    Bewertet die Leistung eines LLMs bei der Extraktion von Anforderungen aus einem Textausschnitt.

    :param predictions: Liste von vom LLM extrahierten Anforderungen.
    :param references: Liste von Listen mit Referenzanforderungen (Gold-Standard + Alternativen).
    :return: Punktzahl für diesen Textausschnitt.
    """
    unmatched_references = set(range(len(references)))  # Indizes der nicht erkannten Referenzen
    unmatched_predictions = set(range(len(predictions)))  # Indizes der nicht zugeordneten Predictions
    matched_references = []
    matched_predictions = []

    # Matching jeder Prediction mit der besten passenden Referenz
    for pred_idx in similarity_scores:
        best_score = 0
        best_ref_idx = None

        for ref_idx in similarity_scores[pred_idx]:
            current_score = similarity_scores[pred_idx][ref_idx]["rouge1"]  # hier wird der Score ausgewählt, z. B. f1

            if current_score > best_score:
                best_score = current_score
                best_ref_idx = ref_idx

        # Falls eine passende Referenz gefunden wurde (z.B. BERTScore ≥ 0.9), markiere sie als erkannt
        if best_ref_idx is not None and best_score >= metric_score_threshold:
            unmatched_references.discard(best_ref_idx)
            matched_references.append(best_ref_idx)
            unmatched_predictions.discard(pred_idx)
            matched_predictions.append(pred_idx)

    mr.print_text_segment_console_output(predictions, references, similarity_scores, unmatched_predictions, unmatched_references, matched_predictions, matched_references)

    # Sonderfall: Platzhalter-Referenz ("The System does not ...") soll in der Auswertung nicht zählen und wird daher entfernt 
    if len(unmatched_references) == 1 and count_valid_references(references) == 0: # wenn Platzhalter vorhanden, dann ist dieser die alleinige Referenz
        unmatched_references.clear()

    # Sonderfall: Platzhalter-Prediction ("The System does not ...") soll in der Auswertung nicht zählen und wird daher entfernt sofern vorhanden.
    # Da der Platzhalter LLM-generiert ist, werden Alternativen ebenfalls entfernt, sofern diese mit einer Platzhalter-Referenz gematched wurden.
    total_predictions = count_valid_predictions(predictions) # Platzhalter zählt nicht und wird abgezogen, wenn dieser mit der exakt definierten Formulierung vom LLM generiert wurde 
    if (total_predictions == 1 and len(matched_predictions) == 1 and count_valid_references(references) == 0): # LLM generierter Platzhalter könnte gematched werden, entspricht aber nicht der exakten Formulierung und muss somit in diesem Schritt rausgerechnet werden
        total_predictions = 0

    return {
        "unmatched_predictions": len(unmatched_predictions),
        "unmatched_references": len(unmatched_references),
        "total_predictions": total_predictions  # gibt die Anzahl aller Predictions zurück, abzüglich (gematched) Platzhalter
    }



def calculate_all_metrics_for_text_segment(predictions: List[str], references: List[List[str]]) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Berechnet BLEU, ROUGE-1 und METEOR für jede Prediction in Bezug auf jede Referenzanforderung.

    :param predictions: Liste der vom LLM erzeugten Anforderungen.
    :param references: Liste von Listen mit Referenzanforderungen (Gold + Alternativen).
    :return: Verschachteltes Dictionary mit Scores für jede Prediction-Referenz-Paarung.
    """
    results_dict = {}

    for pred_idx, prediction in enumerate(predictions):
        results_dict[pred_idx] = {}

        for ref_idx, reference_list in enumerate(references):
            # Berechnungen
            rouge_result = me.calculate_rouge([reference_list], [prediction])
            #bleu_result = me.calculate_bleu([reference_list], [prediction])

            results_dict[pred_idx][ref_idx] = {
                #"meteor": meteor_result["meteor"],
                "rouge1": rouge_result["rouge1"],
                #"bleu": bleu_result["bleu"]
            }

    return results_dict



def split_llm_output_numbered(llm_output: str) -> List[str]:
    """
    Splits a numbered list of requirements from LLM output.
    
    :param llm_output: The string containing numbered requirements.
    :return: A list of individual requirements.
    """
    items = [req.strip() for req in re.split(r'\d+\.\s', llm_output) if req.strip()]
    return items if items else [" "]



def extract_reference_with_placeholder(requirements: dict) -> List[List[str]]:
    """
    Gibt Referenzanforderungen zurück. Falls keine vorhanden sind, wird eine Platzhalteranforderung eingefügt.

    :param requirements: Dictionary der Anforderungen zu einem Textsegment.
    :return: Liste von Listen mit Referenzanforderungen (inkl. Platzhalter bei Bedarf).
    """
    formatted_references = extract_reference(requirements)

    if not any(formatted_references):  # prüft, ob alle inneren Listen leer sind
        return [["This text does not contain any functional requirement information."]]
    
    return formatted_references



def extract_reference(requirements: dict) -> List[List[str]]:
    """
    Formats the reference requirements
    
    :param requirements: Dictionary of requirements for a given text_id
    :return: A nested list with gold references and alternatives
    """
    formatted_references = []
    
    for req_id, req_data in requirements.items():
        references = [req_data["gold_reference"]] if req_data["gold_reference"] else []
        formatted_references.append(references)
    
    return formatted_references



def merge_results_into_cumulative_csv():
    """
    Liest die CSV-Dateien 'cumulative_results_per_textsegment.csv' und 'results_per_textsegment.csv',
    fügt sie zeilenweise zusammen und speichert das Ergebnis als neue 'cumulative_results_per_textsegment.csv'.
    Bereits existierende Daten in der Cumulative-Datei bleiben bestehen, es sei denn,
    Duplikate (identische Zeilen) treten auf.
    """
    cumulative_path = "cumulative_results_per_textsegment.csv"
    current_run_path = "results_per_textsegment.csv"

    # Prüfen, ob beide Dateien existieren
    if not os.path.exists(current_run_path):
        print("❌ Fehler: 'results_per_textsegment.csv' wurde nicht gefunden.")
        return

    # Beide Dateien laden
    df_new = pd.read_csv(current_run_path)

    if os.path.exists(cumulative_path):
        df_cumulative = pd.read_csv(cumulative_path)
        # Zusammenführen (Anhängen) – Duplikate entfernen, falls vorhanden
        combined_df = pd.concat([df_cumulative, df_new], ignore_index=True)
        print("🔁 Bestehende cumulative-Datei wurde ergänzt.")
    else:
        combined_df = df_new
        print("🆕 Neue cumulative-Datei wurde erstellt.")

    # Ergebnis speichern
    combined_df.to_csv(cumulative_path, index=False, encoding="utf-8")
    print(f"✅ Zusammengeführte CSV gespeichert unter: {cumulative_path}")



def export_append_results_per_textsegment_to_csv(results_per_textsegment_of_all_runs: dict):
    """
    Ergänzt das Dictionary results_per_textsegment_of_all_runs zur bestehenden CSV-Datei
    oder erstellt sie neu, falls sie nicht existiert.

    :param results_per_textsegment_of_all_runs: Dictionary mit den Ergebnissen aller Textsegmente
    """
    save_path = "interim_results_per_textsegment.csv"

    if not results_per_textsegment_of_all_runs:
        print("⚠️ Kein Inhalt zum Exportieren vorhanden.")
        return

    # Dictionary in DataFrame umwandeln
    df = pd.DataFrame.from_dict(results_per_textsegment_of_all_runs, orient="index")

    # CSV exportieren
    df.to_csv(save_path, index=False, encoding="utf-8")

    print(f"\n✅ CSV-Datei mit den Ergebnissen gespeichert unter: {save_path}")



def export_results_per_textsegment_to_csv(results_per_textsegment_of_all_runs: dict):
    """
    Exportiert das Dictionary results_per_textsegment_of_all_runs in eine CSV-Datei.

    :param results_per_textsegment_of_all_runs: Dictionary mit den Ergebnissen aller Textsegmente
    :param save_path: Speicherpfad für die CSV-Datei
    """

    save_path = "results_per_textsegment.csv"

    if not results_per_textsegment_of_all_runs:
        print("⚠️ Kein Inhalt zum Exportieren vorhanden.")
        return

    # Dictionary in DataFrame umwandeln
    df = pd.DataFrame.from_dict(results_per_textsegment_of_all_runs, orient="index")

    # CSV exportieren
    df.to_csv(save_path, index=False, encoding="utf-8")

    print(f"\n✅ CSV-Datei mit den Ergebnissen gespeichert unter: {save_path}")



def run_multiple_evaluations(json_file: str, thresholds: List[float], models: List[str], runs_per_model: int) :
    """
    Führt die Evaluation des Datensatzes mehrfach durch.

    :param json_file: Pfad zur JSON-Datei mit den Benchmarkdaten.
    :param runs: Anzahl der Wiederholungen.
    """
    global results_per_textsegment_of_all_runs

    run_id = 1

    for model in models:
        print(f"\n🤖 Starte Runs für Modell: {model}\n")

        for threshold in thresholds:
            print(f"\n🎯 Starte Runs für Threshold {threshold} (Modell: {model}, {runs_per_model} Wiederholungen)\n")

            for m_run in range(runs_per_model):  # Wiederholungen pro Modell
                print(f"\n🚀 Starting evaluation run {run_id} (Modell: {model}, Threshold: {threshold}, Modell-Run {m_run+1})")

                # Datensatz evaluieren
                evaluate_dataset(json_file, run_id=run_id, metric_score_threshold=threshold, model=model)

                # Ergebnisse zusammenfassen
                mr.summarize_single_run_results(run_id, results_per_textsegment_of_all_runs)

                print(f"\n✅ Run {run_id} finished.\n")
                export_append_results_per_textsegment_to_csv(results_per_textsegment_of_all_runs)
                print("-" * 50)
                run_id += 1

    export_results_per_textsegment_to_csv(results_per_textsegment_of_all_runs)
    merge_results_into_cumulative_csv()

    mr.analysis_data(results_per_textsegment_of_all_runs, json_file)



def main():
    """
    Main function to start the evaluation process.
    """
    run_multiple_evaluations(
        json_file="BenchmarkRequirements.json", 
        thresholds=[0.9], 
        #models=["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen3:8b", "qwen3:14b", "qwen3:32b", "gpt-oss:20b", "gpt-oss:120b", "gpt-4", "gpt-5"],
        models=["gpt-4", "gpt-5"],
        runs_per_model=5
    )
    


if __name__ == "__main__":
    main()