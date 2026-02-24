import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import EvaluateJSON
import re

from typing import List, Dict, Set


benchmark_results_each_runs = {} # enthält für jeden run alle Einstellungsparameter (Model, Schwellwert), Kennzahlen (Precision, Recall, F1-Score) und Ansolutzahlen (Pred., Ref., UnMatPred., UnMatRef.)
average_benchmark_results_for_each_model_threshold = {} # enthält für alle Runs mit dem gleichen Schwellwert und gleichem Modell Kennzahlen die Durchschnittswerte der Kennzahlen (Precision, Recall, F1-Score) und Ansolutzahlen (Pred., Ref., UnMatPred., UnMatRef.)
results_per_textsegment_of_all_runs_with_textsegement_analysis_information = {}
benchmark_results_each_runs_with_subgroup_ratio = {}
average_benchmark_results_for_each_model_threshold_with_subgroup_ratio = {}
benchmark_results_each_runs_with_subgroup_textlength = {}
average_benchmark_results_for_each_model_threshold_with_subgroup_textlength = {}


# ------------------------------------------------------- Functions for monitoring the results per text segment during runs --------------------------------------------------------

def print_scores_console_output(score_dict: Dict[int, Dict[int, Dict[str, float]]]):
    """
    Gibt die Ähnlichkeitsmetriken strukturiert in der Konsole aus, ohne explizite Übergabe der Metriknamen.

    :param score_dict: Verschachteltes Dictionary mit Metriken für Prediction-Referenz-Paare.
    """
    print("📊 Ähnlichkeitsmetriken (Prediction ↔ Reference):")
    for pred_idx, ref_scores in score_dict.items():
        for ref_idx, metric_values in ref_scores.items():
            # Dynamisch alle Metrik-Namen und Werte auslesen
            metric_parts = [f"{name.upper()}: {value:.3f}" for name, value in metric_values.items()]
            score_line = f"  Prediction {pred_idx} ↔ Reference {ref_idx} | " + ", ".join(metric_parts)
            print(score_line)
    print()



def print_text_segment_console_output(predictions: List[str], references: List[str], similarity_scores: Dict[int, Dict[int, Dict[str, float]]], unmatched_predictions: Set[int], unmatched_references: Set[int], matched_predictions: Set[int], matched_references: Set[int]):
    """
    Gibt eine übersichtliche Konsolenausgabe zur Bewertung eines Textausschnitts aus.

    :param predictions: Liste der vom LLM extrahierten Anforderungen.
    :param similarity_scores: Verschachteltes Dictionary mit BERTScore-Werten zwischen Predictions und Referenzen.
    :param unmatched_predictions: Indizes der nicht zugeordneten LLM-Anforderungen.
    :param unmatched_references: Indizes der nicht erkannten Referenzanforderungen.
    """

    print("📄 Predictions:")
    for i, pred in enumerate(predictions):
        print(f"  [{i}] {pred}")
    print("📄 References:")
    for i, ref in enumerate(references):
        print(f"  [{i}] {ref}")    
    
    print()
    print_scores_console_output(similarity_scores)
    print()

    print(f"❌ Unmatched Reference Indices: {sorted(unmatched_references)}")
    print(f"❌ Unmatched Prediction Indices: {sorted(unmatched_predictions)}")
    print(f"✅ Matched Reference Indices: {sorted(matched_references)}")
    print(f"✅ Matched Prediction Indices: {sorted(matched_predictions)}")


# ------------------------------------------------------- Functions for monitoring all results after completion of each run --------------------------------------------------------


def summarize_single_run_results(run_id: int, results_per_textsegment_of_all_runs: dict):
    """
    Berechnet die Gesamtmetriken (Precision, Recall, F1) für einen Benchmark-Durchlauf
    und speichert sie unter einer Run-ID im benchmarbenchmark_results_valuesk_run_log.

    :param run_id: Eindeutige ID des Durchlaufs (z. B. "run_1")
    :param threshold: Aktuell verwendeter Ähnlichkeitsschwellenwert
    :param results_per_textsegment_of_all_runs: Dictionary mit Ergebnissen pro Textsegment
    """
    if not results_per_textsegment_of_all_runs:
        print(f"⚠️ Keine Ergebnisse vorhanden für {run_id}.")
        return

    filtered = {k: v for k, v in results_per_textsegment_of_all_runs.items() if v["run_id"] == run_id}
    df = pd.DataFrame.from_dict(filtered, orient="index")

    total_predictions = df["total_predictions"].sum()
    total_references = df["total_references"].sum()
    unmatched_predictions = df["unmatched_predictions"].sum()
    unmatched_references = df["unmatched_references"].sum()

    true_positives = total_predictions - unmatched_predictions

    # Precision: TP / (TP + FP) → entspricht TP / total_predictions
    precision = true_positives / total_predictions if total_predictions > 0 else 0

    # Recall: TP / (TP + FN) → entspricht TP / total_references
    recall = true_positives / total_references if total_references > 0 else 0

    # F1-Score: harmonisches Mittel
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("\n📊 Gesamtergebnis (pro Textsegment): \n")
    print(df)

    print(f"\n📊 Gesamtauswertung (Run: {run_id})\n") 
    print(f"📈 Precision:            {precision:.2%}")
    print(f"📈 Recall:               {recall:.2%}")
    print(f"📈 F1-Score:             {f1:.2%}")
    print(f"🎯 Summe aller Predictions:   {total_predictions}")
    print(f"🎯 Summe aller References:    {total_references}")



# ------------------------------------------------------- Functions for monitoring all results after completion of the runs --------------------------------------------------------


def analysis_data(results_per_textsegment_of_all_runs: dict, json_file: str):

    # Ergebniswerte berechnen
    compute_benchmark_results_for_each_runs(results_per_textsegment_of_all_runs)
    compute_average_results_by_threshold()
    generate_dictionary_results_per_textsegment_of_all_runs_with_textsegement_analysis_information(results_per_textsegment_of_all_runs, json_file)
    compute_benchmark_results_each_textsegment_with_subgroup_ratio()
    compute_average_results_by_threshold_with_subgroup_ratio()
    compute_benchmark_results_each_textsegment_with_subgroup_textlength()
    compute_average_results_by_threshold_with_subgroup_textlength()

    # Diagramme erzeugen
    plot_benchmark_bar_chart_average_precision_recall_f1score_for_all_models()
    plot_benchmark_bar_chart_average_precision_recall_f1score_for_subgroup_ratio_for_all_models()
    plot_benchmark_bar_chart_average_precision_recall_f1score_for_subgroup_textlength_for_all_models()
    plot_line_chart_precision_recall_f1score_over_threshold_per_model()
    plot_benchmark_bar_chart_by_metric_grouped_by_metric()
    plot_benchmark_bar_chart_precision_recall_by_metric_only_precision_recall()
    plot_benchmark_bar_chart_precision_recall_by_reasoning_category()
    plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category() #hier wird immer ein Fehler geworfen, wenn nur ein Modell ausgewertet wird
    plot_boxplot_precision_recall_f1_per_model()
    plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category_from_runs()

















    
    



def compute_benchmark_results_for_each_runs(results_per_textsegment_of_all_runs: dict):
    """
    Aggregiert die Ergebnisse aller Textsegmente pro Run und berechnet Precision, Recall und F1.
    
    :param results_per_textsegment_of_all_runs: Dictionary mit allen Einzelergebnissen pro Textsegment.
    :return: Dictionary benchmark_results_each_runs mit aggregierten Werten pro Run.
    """
    global benchmark_results_each_runs
    benchmark_results_each_runs = {}

    # Alle Run-IDs herausfinden
    run_ids = set(v["run_id"] for v in results_per_textsegment_of_all_runs.values())

    for run_id in run_ids:
        # Filter: nur Einträge für diese Run-ID
        filtered = {k: v for k, v in results_per_textsegment_of_all_runs.items() if v["run_id"] == run_id}
        df = pd.DataFrame.from_dict(filtered, orient="index")

        if df.empty:
            continue

        # Summen über alle Textsegmente des Runs
        total_predictions = df["total_predictions"].sum()
        total_references = df["total_references"].sum()
        unmatched_predictions = df["unmatched_predictions"].sum()
        unmatched_references = df["unmatched_references"].sum()

        # True Positives (TP)
        true_positives = total_predictions - unmatched_predictions

        # Precision
        precision = true_positives / total_predictions if total_predictions > 0 else 0

        # Recall
        recall = true_positives / total_references if total_references > 0 else 0

        # F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        # Modell und Threshold aus dem ersten Eintrag des Runs übernehmen
        first_entry = df.iloc[0]
        model = first_entry["model"]
        threshold = first_entry["metric_score_threshold"]

        # Ergebnisse speichern
        benchmark_results_each_runs[run_id] = {
            "model": model,
            "metric_score_threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_predictions": total_predictions,
            "unmatched_predictions": unmatched_predictions,
            "total_references": total_references,
            "unmatched_references": unmatched_references
        }

    # Konsolenoutput als Tabelle
    df_results = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")
    print("\n📊 Aggregierte Ergebnisse aller Runs:\n")
    print(df_results.round(4))



def compute_average_results_by_threshold():
    """
    Berechnet die Durchschnittswerte aller Runs pro Modell-Threshold-Kombination
    und speichert sie zentral im Dictionary 'average_benchmark_results_for_each_model_threshold'.
    """
    global benchmark_results_each_runs
    global average_benchmark_results_for_each_model_threshold
    average_benchmark_results_for_each_model_threshold = {}  # Reset

    if not benchmark_results_each_runs:
        print("⚠️ Keine aggregierten Ergebnisse pro Run vorhanden.")
        return

    # DataFrame aus den Run-Ergebnissen
    df = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")

    # Gruppierung nach Modell und Threshold
    grouped = df.groupby(["model", "metric_score_threshold"]).mean().reset_index()

    # Dictionary befüllen
    for idx, row in grouped.iterrows():
        average_benchmark_results_for_each_model_threshold[idx + 1] = {
            "model": row["model"],
            "metric_score_threshold": row["metric_score_threshold"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "total_predictions": row["total_predictions"],
            "unmatched_predictions": row["unmatched_predictions"],
            "total_references": row["total_references"],
            "unmatched_references": row["unmatched_references"],
        }

    # Konsolen-Output tabellarisch
    print("\n📊 Durchschnittswerte je Modell-Threshold-Kombination:\n")
    print(pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index").round(4))



def generate_dictionary_results_per_textsegment_of_all_runs_with_textsegement_analysis_information(results_per_textsegment_of_all_runs: dict, json_file: str):
    """
    Kombiniert die Benchmark-Ergebnisse pro Textausschnitt mit zusätzlichen Analyseinformationen
    (aus EvaluateJSON.evaluate_dataset) und speichert sie in einem globalen Dictionary.

    :param results_per_textsegment_of_all_runs: Dictionary mit Ergebnissen pro Run und Textausschnitt
    :param json_file: Pfad zur Benchmark-JSON-Datei
    """

    global results_per_textsegment_of_all_runs_with_textsegement_analysis_information

    # Hole die Analyseinformationen zu allen Textausschnitten
    text_analysis_results = EvaluateJSON.evaluate_dataset(json_file)

    # Neues Dictionary befüllen
    combined = {}
    for key, entry in results_per_textsegment_of_all_runs.items():
        text_id = str(entry["text_id"])  # text_id in Analyse-Dict ist String

        if text_id in text_analysis_results:
            # Kombiniere Benchmark-Ergebnisse mit Textanalyse-Infos
            combined[key] = {**entry, **text_analysis_results[text_id]}
        else:
            # Falls kein Analyse-Eintrag existiert, nur die Benchmark-Daten übernehmen
            combined[key] = entry

    # Als Klassenvariable (global) speichern
    results_per_textsegment_of_all_runs_with_textsegement_analysis_information = combined

    # In DataFrame konvertieren
    df = pd.DataFrame.from_dict(results_per_textsegment_of_all_runs_with_textsegement_analysis_information, orient="index")

    # Tabellarisch ausgeben
    print("\n📊 Ergebnisse mit Textanalyse-Informationen kombiniert:\n")
    print(df)



def compute_benchmark_results_each_textsegment_with_subgroup_ratio():
    """
    Aggregiert die Ergebnisse aller Textsegmente pro Run und Subgruppe (basierend auf ratio_reference_to_total)
    und berechnet Precision, Recall und F1.

    :param results_per_textsegment_of_all_runs_with_textsegement_analysis_information: 
           Dictionary mit allen Einzelergebnissen + Textanalyseinfos pro Textsegment.
    """
    global results_per_textsegment_of_all_runs_with_textsegement_analysis_information
    global benchmark_results_each_runs_with_subgroup_ratio
    benchmark_results_each_runs_with_subgroup_ratio = {}

    # Alle Run-IDs herausfinden
    run_ids = set(v["run_id"] for v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.values())

    for run_id in run_ids:
        # Filter: nur Einträge für diese Run-ID
        filtered = {
            k: v for k, v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.items()
            if v["run_id"] == run_id
        }
        df = pd.DataFrame.from_dict(filtered, orient="index")

        if df.empty:
            continue

        # Modell und Threshold aus erstem Eintrag übernehmen
        first_entry = df.iloc[0]
        model = first_entry["model"]
        threshold = first_entry["metric_score_threshold"]

        # Subgruppen definieren
        subgroup_conditions = {
            "full_relevant": df["ratio_reference_to_total"] == 1.0,
            "full_irrelevant": df["ratio_reference_to_total"] == 0.0,
            "mixed": (df["ratio_reference_to_total"] > 0.0) & (df["ratio_reference_to_total"] < 1.0),
        }

        for subgroup_name, condition in subgroup_conditions.items():
            subgroup_df = df[condition]

            if subgroup_df.empty:
                continue  # Keine Einträge in dieser Subgruppe für diesen Run

            num_text_segments = len(subgroup_df)

            # Summen über alle Textsegmente der Subgruppe
            total_predictions = subgroup_df["total_predictions"].sum()
            total_references = subgroup_df["total_references"].sum()
            unmatched_predictions = subgroup_df["unmatched_predictions"].sum()
            unmatched_references = subgroup_df["unmatched_references"].sum()

            true_positives = total_predictions - unmatched_predictions

            precision = true_positives / total_predictions if total_predictions > 0 else 0
            recall = true_positives / total_references if total_references > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            # Ergebnisse speichern: Key = Kombination aus run_id + Subgruppe
            benchmark_results_each_runs_with_subgroup_ratio[f"{run_id}_{subgroup_name}"] = {
                "run_id": run_id,
                "model": model,
                "metric_score_threshold": threshold,
                "subgroup": subgroup_name,
                "num_text_segments": num_text_segments,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "total_predictions": total_predictions,
                "unmatched_predictions": unmatched_predictions,
                "total_references": total_references,
                "unmatched_references": unmatched_references
            }

    # Ausgabe als Tabelle
    df_results = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_ratio, orient="index")
    print("\n📊 Aggregierte Ergebnisse nach Relevanz-Subgruppen pro Run:\n")
    print(df_results.round(4))



def compute_average_results_by_threshold_with_subgroup_ratio():
    """
    Berechnet die Durchschnittswerte aller Subgruppen-Ergebnisse pro Modell-Threshold-Subgroup-Kombination.
    Speichert sie zentral im Dictionary 'average_benchmark_results_for_each_model_threshold_with_subgroup_ratio'.
    """
    global benchmark_results_each_runs_with_subgroup_ratio
    global average_benchmark_results_for_each_model_threshold_with_subgroup_ratio
    average_benchmark_results_for_each_model_threshold_with_subgroup_ratio = {}  # Reset

    if not benchmark_results_each_runs_with_subgroup_ratio:
        print("⚠️ Keine Subgruppen-Ergebnisse vorhanden.")
        return

    # DataFrame erstellen
    df = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_ratio, orient="index")

    # Gruppieren nach Modell, Threshold und Subgruppe
    grouped = df.groupby(["model", "metric_score_threshold", "subgroup"]).mean().reset_index()

    # Dictionary befüllen
    for idx, row in grouped.iterrows():
        average_benchmark_results_for_each_model_threshold_with_subgroup_ratio[idx + 1] = {
            "model": row["model"],
            "metric_score_threshold": row["metric_score_threshold"],
            "subgroup": row["subgroup"],
            "num_text_segments": int(row["num_text_segments"]),
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "total_predictions": row["total_predictions"],
            "unmatched_predictions": row["unmatched_predictions"],
            "total_references": row["total_references"],
            "unmatched_references": row["unmatched_references"],
        }

    # Ausgabe in Konsole
    print("\n📊 Durchschnittswerte je Modell-Threshold-Relevanz-Subgruppe:\n")
    print(pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold_with_subgroup_ratio, orient="index").round(4))


def compute_benchmark_results_each_textsegment_with_subgroup_textlength():
    """
    Aggregiert die Ergebnisse aller Textsegmente pro Run und Subgruppe (basierend auf Textlänge: num_words)
    und berechnet Precision, Recall und F1.

    Subgruppen:
    - short: 0–39 Wörter
    - medium: 40–59 Wörter
    - long: >=60 Wörter
    """
    global results_per_textsegment_of_all_runs_with_textsegement_analysis_information
    global benchmark_results_each_runs_with_subgroup_textlength
    benchmark_results_each_runs_with_subgroup_textlength = {}

    # Alle Run-IDs herausfinden
    run_ids = set(v["run_id"] for v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.values())

    for run_id in run_ids:
        # Filter: nur Einträge für diese Run-ID
        filtered = {
            k: v for k, v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.items()
            if v["run_id"] == run_id
        }
        df = pd.DataFrame.from_dict(filtered, orient="index")

        if df.empty:
            continue

        # Modell und Threshold aus erstem Eintrag übernehmen
        first_entry = df.iloc[0]
        model = first_entry["model"]
        threshold = first_entry["metric_score_threshold"]

        # Subgruppen definieren
        subgroup_conditions = {
            "short": df["num_words"] <= 39,
            "medium": (df["num_words"] >= 40) & (df["num_words"] <= 59),
            "long": df["num_words"] >= 60,
        }

        for subgroup_name, condition in subgroup_conditions.items():
            subgroup_df = df[condition]

            if subgroup_df.empty:
                continue  # Keine Einträge in dieser Subgruppe für diesen Run

            num_text_segments = len(subgroup_df)

            # Summen über alle Textsegmente der Subgruppe
            total_predictions = subgroup_df["total_predictions"].sum()
            total_references = subgroup_df["total_references"].sum()
            unmatched_predictions = subgroup_df["unmatched_predictions"].sum()
            unmatched_references = subgroup_df["unmatched_references"].sum()

            # True Positives (TP)
            true_positives = total_predictions - unmatched_predictions

            # Precision, Recall, F1
            precision = true_positives / total_predictions if total_predictions > 0 else 0
            recall = true_positives / total_references if total_references > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            # Ergebnisse speichern: Key = Kombination aus run_id + Subgruppe
            benchmark_results_each_runs_with_subgroup_textlength[f"{run_id}_{subgroup_name}"] = {
                "run_id": run_id,
                "model": model,
                "metric_score_threshold": threshold,
                "subgroup": subgroup_name,
                "num_text_segments": num_text_segments,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "total_predictions": total_predictions,
                "unmatched_predictions": unmatched_predictions,
                "total_references": total_references,
                "unmatched_references": unmatched_references
            }

    # Ausgabe als Tabelle
    df_results = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_textlength, orient="index")
    print("\n📊 Aggregierte Ergebnisse nach Textlängen-Subgruppen pro Run:\n")
    print(df_results.round(4))


def compute_average_results_by_threshold_with_subgroup_textlength():
    """
    Berechnet die Durchschnittswerte aller Subgruppen-Ergebnisse pro Modell-Threshold-Textlength-Subgruppe.
    Speichert sie zentral im Dictionary 'average_benchmark_results_for_each_model_threshold_with_subgroup_textlength'.
    """
    global benchmark_results_each_runs_with_subgroup_textlength
    global average_benchmark_results_for_each_model_threshold_with_subgroup_textlength
    average_benchmark_results_for_each_model_threshold_with_subgroup_textlength = {}  # Reset

    if not benchmark_results_each_runs_with_subgroup_textlength:
        print("⚠️ Keine Textlängen-Subgruppen-Ergebnisse vorhanden.")
        return

    # DataFrame erstellen
    df = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_textlength, orient="index")

    # Gruppieren nach Modell, Threshold und Subgruppe
    grouped = df.groupby(["model", "metric_score_threshold", "subgroup"]).mean().reset_index()

    # Dictionary befüllen
    for idx, row in grouped.iterrows():
        average_benchmark_results_for_each_model_threshold_with_subgroup_textlength[idx + 1] = {
            "model": row["model"],
            "metric_score_threshold": row["metric_score_threshold"],
            "subgroup": row["subgroup"],
            "num_text_segments": int(row["num_text_segments"]),
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "total_predictions": row["total_predictions"],
            "unmatched_predictions": row["unmatched_predictions"],
            "total_references": row["total_references"],
            "unmatched_references": row["unmatched_references"],
        }

    # Ausgabe in Konsole
    print("\n📊 Durchschnittswerte je Modell-Threshold-Textlängen-Subgruppe:\n")
    print(pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold_with_subgroup_textlength,
                                 orient="index").round(4))
























def plot_benchmark_bar_chart_average_precision_recall_f1score_for_all_models():
    
    save_path="benchmark_bar_chart_paper_style.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX aktivieren (mit Helvetica)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    metrics = ["precision", "recall", "f1_score"]
    df_plot = df[["model", "metric_score_threshold"] + metrics]

    x = np.arange(len(df_plot))  # Positionen für Gruppen
    bar_width = 0.25
    spacing = bar_width * 1.1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Farben definieren: Automatisch drei Farben aus der "Blues"-Colormap holen
    cmap = plt.get_cmap("Purples")  # Möglichkeiten: "Blues", "Greys", "Purples", "Oranges", etc.
    colors = {
        "precision": mcolors.to_hex(cmap(0.4)),  # heller Blauton
        "recall": mcolors.to_hex(cmap(0.6)),     # mittlerer Blauton
        "f1_score": mcolors.to_hex(cmap(0.8))    # dunkler Blauton
    }

    # Balken zeichnen
    bars = {}
    for idx, metric in enumerate(metrics):
        positions = x + (idx - 1) * spacing  # leicht versetzt pro Metrik
        bars[metric] = ax.bar(
            positions,
            df_plot[metric],
            width=bar_width,
            label=metric.capitalize(),
            color=colors[metric]
        )
        # Werte über den Balken anzeigen
        for rect in bars[metric]:
            height = rect.get_height()
            ax.annotate(
                #f"{height:.2f}",
                rf"\textsf{{{height:.2f}}}",  # serifenlose Schriftart (Helvetica) im LaTeX-Stil
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", 
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    # X-Achsenbeschriftung
    # labels = [f"{row['model']} (thr={row['metric_score_threshold']})" for _, row in df_plot.iterrows()] # mit Schwellwert
    labels = df_plot["model"].tolist() # ohne Schwellwert
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)

    # Legende oberhalb, mittig – mit Rahmen (Kasten)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,  # <--- aktiviert Kasten um die Legende
        edgecolor="black",  # optional: Farbe des Rahmens
        fancybox=False,       # optional: eckiger Rahmen statt abgerundet
        framealpha=1.0,     # Transparenz des Kastens (1.0 = undurchsichtig)
        fontsize=10         # Schriftgröße der Legende
    )

    # Diagramm mit allen Rahmenlinien
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))  # verhindert automatisches Überschreiten von 1.0
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Achsen-Ticks-Stil (Konsistenz und Helvetica sicherstellen)
    ax.tick_params(left=False, bottom=False, labelsize=10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)  # Sicherstellen, dass keine Warnung ausgelöst wird
    # ax.yaxis.set_ticks([])  # Y-Achse entfernen, wenn Code ausgeführt wird
    ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in yticks], fontsize=10)

    # Gitterlinie nur auf Y-Achse
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Layout und Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_paper_style.pdf")
    #plt.show()

    print(f"\n✅ Balkendiagramm gespeichert unter: {save_path}")



def plot_benchmark_bar_chart_average_precision_recall_f1score_for_subgroup_ratio_for_all_models():

    save_path = "benchmark_bar_chart_subgroup_ratio.png"
    global average_benchmark_results_for_each_model_threshold_with_subgroup_ratio

    # LaTeX-Stil (Helvetica) aktivieren
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold_with_subgroup_ratio, orient="index")
    metrics = ["precision", "recall", "f1_score"]
    df_plot = df[["model", "subgroup"] + metrics]

    # Gruppierung: Modell + Subgruppe → für X-Achse
    labels = [f"{row['model']} \n({row['subgroup']})" for _, row in df_plot.iterrows()]
    x = np.arange(len(labels))
    bar_width = 0.25
    spacing = bar_width * 1.1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Farben aus Colormap "Purples"
    cmap = plt.get_cmap("Purples")
    colors = {
        "precision": mcolors.to_hex(cmap(0.4)),
        "recall": mcolors.to_hex(cmap(0.6)),
        "f1_score": mcolors.to_hex(cmap(0.8))
    }

    # Balken zeichnen
    bars = {}
    for idx, metric in enumerate(metrics):
        positions = x + (idx - 1) * spacing
        bars[metric] = ax.bar(
            positions,
            df_plot[metric],
            width=bar_width,
            label=metric.capitalize(),
            color=colors[metric]
        )
        # Beschriftung über Balken
        for rect in bars[metric]:
            height = rect.get_height()
            ax.annotate(
                rf"\textsf{{{height:.2f}}}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", 
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    # X-Achsenbeschriftungen
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)

    # Legende oberhalb mit Rahmen
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=10
    )

    # Y-Achse: 0 bis 1.05, saubere Labels
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Achsen-Ticks-Stil (Konsistenz und Helvetica sicherstellen)
    ax.tick_params(left=False, bottom=False, labelsize=10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in yticks], fontsize=10)

    # Gitterlinie Y-Achse
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_subgroup_ratio.pdf")
    # plt.show()

    print(f"\n✅ Subgruppe_Ratio-Balkendiagramm gespeichert unter: {save_path}")



def plot_benchmark_bar_chart_average_precision_recall_f1score_for_subgroup_textlength_for_all_models():

    save_path = "benchmark_bar_chart_subgroup_textlength.png"
    global average_benchmark_results_for_each_model_threshold_with_subgroup_textlength

    # LaTeX-Stil (Helvetica) aktivieren
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold_with_subgroup_textlength, orient="index")
    metrics = ["precision", "recall", "f1_score"]
    df_plot = df[["model", "subgroup"] + metrics]

    # Gruppierung: Modell + Subgruppe → für X-Achse
    labels = [f"{row['model']} \n({row['subgroup']})" for _, row in df_plot.iterrows()]
    x = np.arange(len(labels))
    bar_width = 0.25
    spacing = bar_width * 1.1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Farben aus Colormap "Purples"
    cmap = plt.get_cmap("Purples")
    colors = {
        "precision": mcolors.to_hex(cmap(0.4)),
        "recall": mcolors.to_hex(cmap(0.6)),
        "f1_score": mcolors.to_hex(cmap(0.8))
    }

    # Balken zeichnen
    bars = {}
    for idx, metric in enumerate(metrics):
        positions = x + (idx - 1) * spacing
        bars[metric] = ax.bar(
            positions,
            df_plot[metric],
            width=bar_width,
            label=metric.capitalize(),
            color=colors[metric]
        )
        # Beschriftung über Balken
        for rect in bars[metric]:
            height = rect.get_height()
            ax.annotate(
                rf"\textsf{{{height:.2f}}}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", 
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    # X-Achsenbeschriftungen
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)

    # Legende oberhalb mit Rahmen
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=10
    )

    # Y-Achse: 0 bis 1.05, saubere Labels
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Achsen-Ticks-Stil (Konsistenz und Helvetica sicherstellen)
    ax.tick_params(left=False, bottom=False, labelsize=10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in yticks], fontsize=10)

    # Gitterlinie Y-Achse
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_subgroup_textlength.pdf")
    # plt.show()

    print(f"\n✅ Subgruppe_Textlength-Balkendiagramm gespeichert unter: {save_path}")



def plot_line_chart_precision_recall_f1score_over_threshold_per_model():
    
    save_path = "line_chart_precision_recall_f1score_over_threshold_per_model.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil (Helvetica) aktivieren
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    df = df.sort_values(by=["model", "metric_score_threshold"])
    metrics = ["precision", "recall", "f1_score"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Farbzuweisung pro Metrik (dunklere Grautöne für bessere Lesbarkeit)
    cmap = plt.get_cmap("Purples")
    color_map = {
        "precision": mcolors.to_hex(cmap(0.6)),
        "recall": mcolors.to_hex(cmap(0.75)),
        "f1_score": mcolors.to_hex(cmap(0.9))
    }

    # Alle Modelle ermitteln
    models = df["model"].unique()

    # Liniendiagramme für jede Metrik & jedes Modell
    for model in models:
        df_model = df[df["model"] == model]
        df_model = df_model[df_model["metric_score_threshold"] >= 0.75] #filtert den Wertebereich ab 0.70
        thresholds = df_model["metric_score_threshold"]

        for metric in metrics:
            scores = df_model[metric]
            ax.plot(
                thresholds,
                scores,
                marker="o",
                linestyle="-",
                linewidth=2,
                label=f"{model} {metric.capitalize()}",
                color=color_map[metric]
            )

    # X-Achse: Einheitlich und typografisch konsistent
    x_ticks = np.arange(0.75, 1.01, 0.05)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in x_ticks], fontsize=32)

    # Y-Achse: 0–1.05, typografisch konsistent
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, 1.0, 6)], fontsize=32)

    # ax.set_yticks(np.linspace(0.5, 1.0, 3))
    # ax.set_yticks(np.linspace(0.5, 1.0, 3))
    # ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in np.linspace(0.5, 1.0, 3)], fontsize=18)


    # Achsentitel entfernen
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Rahmenlinien
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Achsen-Ticks-Stil
    ax.tick_params(left=False, bottom=False, labelsize=32)

    # Gitterlinie Y-Achse
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Legende im gewohnten Stil
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.47, 1.22),
        ncol=3,
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        columnspacing=1.5,
        handlelength=1.0,
        handletextpad=0.4,
        fontsize=32
    )

    # Export als Vektor
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("line_chart_precision_recall_f1score_over_threshold_per_model.pdf")
    # plt.show()

    print(f"\n✅ Liniendiagramm gespeichert unter: {save_path}")



def plot_benchmark_bar_chart_by_metric_grouped_by_metric():
    save_path = "benchmark_bar_chart_grouped_by_metric.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil (Helvetica) aktivieren
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    metrics = ["precision", "recall", "f1_score"]
    models = df["model"].unique().tolist()

    # Neue Struktur: X-Achse = Metriken, Balken = Modelle
    x = np.arange(len(metrics))  # [0, 1, 2] für 3 Metriken
    bar_width = 0.8 / len(models)  # Platz je Modell in jeder Metrikgruppe
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    # Modelle ermitteln
    models = df["model"].unique()
    num_models = len(models)

    # Farbintervall manuell festlegen: vermeide zu helle Farben (<0.3)
    start, stop = 0.3, 0.9
    cmap = plt.get_cmap("Purples")
    if num_models > 1:
        colors = [
            cmap(start + (stop - start) * (i / (num_models - 1)))
            for i in range(num_models)
        ]
    else:
        colors = [cmap(0.6)]  # wenn nur ein Modell vorhanden ist, z.B. mittlerer Ton


    # Farben aus Colormap (eine andere z.B. "Purples")
    # cmap = plt.get_cmap("Purples", len(models))  # abgestufte Farben für Modelle
    
    for i, model in enumerate(models):
        model_data = df[df["model"] == model].iloc[0]  # eine Zeile je Modell
        values = [model_data[metric] for metric in metrics]
        positions = x + (i - (len(models) - 1) / 2) * bar_width + (spacing / 2) * (i - 1)
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=model,
            color=colors[i]
        )

        # Werte oberhalb der Balken
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                rf"\textsf{{{height:.2f}}}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    # X-Achse: Metriken
    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"\textsf{{{m.capitalize()}}}" for m in metrics],
        fontsize=10,
        ha="center"
    )

    # Legende: Modelle (wie gewünscht)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(models),
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=10
    )

    # Diagramm-Aussehen
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Y-Achse Helvetica
    ax.tick_params(left=False, bottom=False, labelsize=10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in yticks], fontsize=10)

    # Gitterlinie
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_grouped_by_metric.pdf")
    # plt.show()

    print(f"\n✅ Balkendiagramm (nach Metrik gruppiert) gespeichert unter: {save_path}")


def plot_benchmark_bar_chart_precision_recall_by_metric_only_precision_recall():
    save_path = "benchmark_bar_chart_precision_recall_grouped_by_metric.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil (Helvetica) aktivieren
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    metrics = ["precision", "recall"]  # ⬅️ Nur diese beiden Metriken
    models = df["model"].unique().tolist()

    # X-Achse = Metriken, Balken = Modelle
    x = np.arange(len(metrics))  # [0, 1]
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    # Farbintervall manuell festlegen
    start, stop = 0.3, 0.9
    cmap = plt.get_cmap("Purples")
    num_models = len(models)

    if num_models > 1:
        colors = [
            cmap(start + (stop - start) * (i / (num_models - 1)))
            for i in range(num_models)
        ]
    else:
        colors = [cmap(0.6)]

    # Balken zeichnen
    for i, model in enumerate(models):
        model_data = df[df["model"] == model].iloc[0]
        values = [model_data[metric] for metric in metrics]
        positions = x + (i - (num_models - 1) / 2) * bar_width + (spacing / 2) * (i - 1)
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=model,
            color=colors[i]
        )

        # Werte oberhalb der Balken
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                rf"\textsf{{{height:.2f}}}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    # X-Achse: Beschriftung
    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"\textsf{{{m.capitalize()}}}" for m in metrics],
        fontsize=10,
        ha="center"
    )

    # Legende: Modellnamen
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(models),
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=10
    )

    # Diagramm-Aussehen
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Y-Achse Helvetica
    ax.tick_params(left=False, bottom=False, labelsize=10)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in yticks], fontsize=10)

    # Gitterlinie
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_precision_recall_grouped_by_metric.pdf")
    # plt.show()

    print(f"\n✅ Balkendiagramm (nur Precision & Recall) gespeichert unter: {save_path}")


def plot_benchmark_bar_chart_precision_recall_by_reasoning_category():
    save_path = "benchmark_bar_chart_precision_recall_reasoning_split.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil aktivieren (Helvetica)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    metrics = ["precision", "recall"]
    models = df["model"].tolist()
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    # Modell-Gruppen definieren
    reasoning_models = ["gpt-5", "gpt-oss:120b", "gpt-oss:20b", "qwen3:8b", "qwen3:14b", "qwen3:32b"]
    non_reasoning_models = ["gpt-4", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"]

    # Farbabstufungen vorbereiten
    cmap_reasoning = plt.get_cmap("Greys")
    cmap_non_reasoning = plt.get_cmap("Oranges")
    num_reasoning = len(reasoning_models)
    num_non_reasoning = len(non_reasoning_models)

    colors_reasoning = {
        model: cmap_reasoning(0.3 + 0.6 * i / max(num_reasoning - 1, 1))
        for i, model in enumerate(reasoning_models)
    }

    colors_non_reasoning = {
        model: cmap_non_reasoning(0.3 + 0.6 * i / max(num_non_reasoning - 1, 1))
        for i, model in enumerate(non_reasoning_models)
    }

    # Alle Farben zusammenführen
    all_colors = {**colors_reasoning, **colors_non_reasoning}

    for i, model in enumerate(models):
        model_data = df[df["model"] == model].iloc[0]
        values = [model_data[metric] for metric in metrics]
        positions = x + (i - (len(models) - 1) / 2) * bar_width + (spacing / 2) * (i - 1)

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=model,
            color=all_colors.get(model, "#999999")  # Fallback-Farbe
        )

        # Werte oberhalb der Balken
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                rf"\textsf{{{height:.2f}}}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    # Achsen und Legende
    ax.set_xticks(x)
    ax.set_xticklabels(
        [rf"\textsf{{{m.capitalize()}}}" for m in metrics],
        fontsize=10,
        ha="center"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(models),
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=10
    )

    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.tick_params(left=False, bottom=False, labelsize=10)
    yticks = ax.get_yticks()
    ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in yticks], fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_precision_recall_reasoning_split.pdf")

    print(f"\n✅ Balkendiagramm (Precision & Recall, Reasoning vs. Non-Reasoning Modelle) gespeichert unter: {save_path}")


def plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category():
    save_path = "benchmark_bar_chart_precision_recall_f1score_reasoning_split.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil aktivieren (Helvetica)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    metrics = ["precision", "recall", "f1_score"]
    models = df["model"].tolist()
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    # Modelle sortiert nach Parameterrangfolge
    reasoning_models = ["qwen3:8b", "qwen3:14b", "gpt-oss:20b", "qwen3:32b", "gpt-oss:120b", "gpt-5"]
    non_reasoning_models = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "gpt-4"]
    #models = non_reasoning_models + reasoning_models  # gewünschte Reihenfolge

    models_ordered = non_reasoning_models + reasoning_models
    # Nur Modelle verwenden, die tatsächlich im DataFrame existieren
    models = [m for m in models_ordered if m in df["model"].values]

    # Farbabstufungen vorbereiten
    cmap_reasoning = plt.get_cmap("Greens")
    cmap_non_reasoning = plt.get_cmap("Purples")

    colors_reasoning = {
        model: cmap_reasoning(0.3 + 0.45 * i / max(len(reasoning_models) - 1, 1))
        for i, model in enumerate(reasoning_models)
    }

    colors_non_reasoning = {
        model: cmap_non_reasoning(0.3 + 0.45 * i / max(len(non_reasoning_models) - 1, 1))
        for i, model in enumerate(non_reasoning_models)
    }

    # Alle Farben zusammenführen
    all_colors = {**colors_reasoning, **colors_non_reasoning}

    for i, model in enumerate(models):
        model_data = df[df["model"] == model].iloc[0]
        values = [model_data[metric] for metric in metrics]
        positions = x + (i - (len(models) - 1) / 2) * bar_width + (spacing / 2) * (i - 1)

        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=model,
            color=all_colors.get(model, "#999999")  # Fallback-Farbe
        )

        # Werte oberhalb der Balken
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    # Achsen und Legende
    ax.set_xticks(x)
    ax.set_xticklabels(
        # [rf"\textsf{{{m.capitalize()}}}" for m in metrics],
        [rf"{m.capitalize()}" for m in metrics],
        fontsize=23,
        # [r"\normalsize\textsf{{{}}}".format(m.capitalize()) for m in metrics],
        # fontsize=14,
        ha="center"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, 1.25),
        ncol=5,
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=23,
        columnspacing=0.7,
        handlelength=1.25,      # Länge des Farbkästchens (kleiner = kompakter)
        handletextpad=0.2      # Abstand zwischen Farbkästchen und Text reduzieren
    )

    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.tick_params(left=False, bottom=False, labelsize=23)
    yticks = ax.get_yticks()
    ax.set_yticklabels(
        [f"{t:.1f}" for t in yticks], 
        fontsize=23
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_precision_recall_f1score_reasoning_split.pdf")

    print(f"\n✅ Balkendiagramm (Precision & Recall, Reasoning vs. Non-Reasoning Modelle) gespeichert unter: {save_path}")


def plot_boxplot_precision_recall_f1_per_model():
    """
    Erstellt ein Boxplot-Diagramm der Verteilungen von Precision, Recall und F1-Score
    für jedes Modell basierend auf allen Einzel-Runs.
    """
    global benchmark_results_each_runs

    if not benchmark_results_each_runs:
        print("⚠️ Keine Benchmark-Ergebnisse verfügbar.")
        return

    # LaTeX Schriftstil setzen (Computer Modern Roman)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # DataFrame erstellen
    df = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")

    # Daten umstrukturieren (Long-Format für seaborn)
    df_long = pd.melt(
        df,
        id_vars=["model"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score"
    )

    # Plot
    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=df_long,
        x="metric",
        y="score",
        hue="model",
        width=0.6,
        fliersize=2,  # Größe der Ausreißer
        linewidth=1
    )

    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=11)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig("boxplot_precision_recall_f1_per_model.png", dpi=300)
    plt.savefig("boxplot_precision_recall_f1_per_model.pdf")
    #plt.show()

    print("\n✅ Boxplot Diagramm gespeichert unter: boxplot_precision_recall_f1_per_model.*")


def plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category_from_runs():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pandas as pd

    save_path = "benchmark_bar_chart_precision_recall_f1score_reasoning_split_from_runs.png"
    global benchmark_results_each_runs

    # LaTeX-Stil aktivieren (Computer Modern)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # Daten vorbereiten
    df = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")

    # Gruppieren nach Modell
    grouped = df.groupby("model")[["precision", "recall", "f1_score"]]
    summary = grouped.agg(["mean", "min", "max"])  # Erweiterbar: , "min", "max"
    #summary.columns = summary.columns.droplevel(1)  # Mehrstufigen Index flach machen

    metrics = ["precision", "recall", "f1_score"]
    models = summary.index.tolist()
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    # Modelle sortiert nach Parameterrangfolge
    reasoning_models = ["qwen3:8b", "qwen3:14b", "gpt-oss:20b", "qwen3:32b", "gpt-oss:120b", "gpt-5"]
    non_reasoning_models = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "gpt-4"]
    models_ordered = non_reasoning_models + reasoning_models
    summary = summary.loc[[m for m in models_ordered if m in summary.index]]  # Filter & Sortierung

    # Farbabstufungen vorbereiten
    cmap_reasoning = plt.get_cmap("Greens")
    cmap_non_reasoning = plt.get_cmap("Purples")

    colors_reasoning = {
        model: cmap_reasoning(0.3 + 0.45 * i / max(len(reasoning_models) - 1, 1))
        for i, model in enumerate(reasoning_models)
    }

    colors_non_reasoning = {
        model: cmap_non_reasoning(0.3 + 0.45 * i / max(len(non_reasoning_models) - 1, 1))
        for i, model in enumerate(non_reasoning_models)
    }

    all_colors = {**colors_reasoning, **colors_non_reasoning}

    for i, model in enumerate(summary.index):
        mean_values = [summary.loc[model][(metric, "mean")] for metric in metrics]
        min_values  = [summary.loc[model][(metric, "min")]  for metric in metrics]
        max_values  = [summary.loc[model][(metric, "max")]  for metric in metrics]

        neg_errors = np.array(mean_values) - np.array(min_values)
        pos_errors = np.array(max_values) - np.array(mean_values)

        yerr = np.vstack([neg_errors, pos_errors])

        positions = x + (i - (len(summary.index) - 1) / 2) * bar_width + (spacing / 2) * (i - 1)

        bars = ax.bar(
            positions,
            mean_values,
            width=bar_width,
            label=model,
            color=all_colors.get(model, "#999999"),
            yerr = yerr,
            error_kw=dict(elinewidth=0.8, capsize=4, ecolor="black")
        )

        # Werte oberhalb der Balken
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold"
            )

    # Achsen & Legende
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m.capitalize()}" for m in metrics],
        fontsize=18,
        ha="center"
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, 1.25),
        ncol=5,
        frameon=False,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        fontsize=18,
        columnspacing=0.7,
        handlelength=1.25,
        handletextpad=0.2
    )

    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, 1.0, 6)], fontsize=23)
    ax.tick_params(left=False, bottom=False, labelsize=18)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Export
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("benchmark_bar_chart_precision_recall_f1score_reasoning_split_from_runs.pdf")

    print(f"\n✅ Balkendiagramm (Precision, Recall, F1 über Runs) gespeichert unter: {save_path}")











def load_csv_and_renumber_run_ids(csv_path, chunk_size):
    """
    Liest eine CSV-Datei ein, speichert sie als Dictionary und passt die run_id basierend auf Blöcken an.

    :param csv_path: Pfad zur CSV-Datei
    :param chunk_size: Anzahl der Zeilen pro run_id
    :return: Dictionary mit neu zugewiesenen run_ids
    """
    # CSV einlesen (Reihenfolge bleibt erhalten)
    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠️ CSV-Datei ist leer.")
        return {}

    # Neue run_ids berechnen: 120 Zeilen pro Block
    df["run_id"] = [(i // chunk_size) + 1 for i in range(len(df))]

    # In Dictionary umwandeln
    results_dict = df.to_dict(orient="index")

    # Kontroll-Export basierend auf dem Dictionary
    df_control = pd.DataFrame.from_dict(results_dict, orient="index")
    df_control.to_csv("controll_run_id_adaption.csv", index=False, encoding="utf-8")
    print("📁 Kontroll-Export gespeichert unter: controll_run_id_adaption.csv")

    print(f"✅ CSV geladen und run_ids neu nummeriert ({chunk_size} Textausschnitte pro Run).")
    return results_dict



def load_results_dict_from_csv(csv_path: str) -> dict:
    """
    Lädt eine CSV-Datei ein und stellt das Dictionary results_per_textsegment_of_all_runs wieder her.
    
    :param csv_path: Pfad zur CSV-Datei
    :return: Dictionary mit Ergebnissen pro Textsegment
    """
    df = pd.read_csv(csv_path)

    # Sicherstellen, dass numerische Felder korrekt sind (optional: konvertieren)
    df = df.convert_dtypes()

    # Umwandeln in Dictionary mit int-Key
    results_dict = {
        idx: row._asdict() if hasattr(row, "_asdict") else row.to_dict()
        for idx, row in df.iterrows()
    }

    print(f"\n✅ CSV-Datei erfolgreich geladen: {csv_path}")
    return results_dict



def main():

    #csv_file = "results_per_textsegment.csv"  # Adjust the path if necessary
    csv_file = "cumulative_results_per_textsegment.csv"  # Adjust the path if necessary
    #csv_file = "cumulative_results_per_textsegment_final_version.csv"
    json_file = "BenchmarkRequirements.json"

    #results_per_textsegment_of_all_runs = load_results_dict_from_csv(csv_file)
    results_per_textsegment_of_all_runs = load_csv_and_renumber_run_ids(csv_file, 120)
    #results_per_textsegment_of_all_runs = load_csv_and_renumber_run_ids("results_per_textsegment_analysis_threshold.csv", 60)

    analysis_data(results_per_textsegment_of_all_runs, json_file)



if __name__ == "__main__":
    main()    


