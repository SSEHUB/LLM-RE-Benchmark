import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import EvaluateJSON
import re

from typing import List, Dict, Set


benchmark_results_each_runs = {} # Contains all setting parameters (model, threshold value), key figures (precision, recall, F1 score), and absolute numbers (pred., ref., unmatpred., unmatref.) for each run.
average_benchmark_results_for_each_model_threshold = {} # Contains key figures for all runs with the same threshold value and model, including the average values of the key figures (precision, recall, F1 score) and absolute values (pred., ref., unmatpred., unmatref.).
results_per_textsegment_of_all_runs_with_textsegement_analysis_information = {}
benchmark_results_each_runs_with_subgroup_ratio = {}
average_benchmark_results_for_each_model_threshold_with_subgroup_ratio = {}
benchmark_results_each_runs_with_subgroup_textlength = {}
average_benchmark_results_for_each_model_threshold_with_subgroup_textlength = {}


# ------------------------------------------------------- Functions for monitoring the results per text segment during runs --------------------------------------------------------

def print_scores_console_output(score_dict: Dict[int, Dict[int, Dict[str, float]]]):
    """
    Outputs the similarity metrics in a structured format in the console without explicitly passing the metric names.

    :param score_dict: Nested dictionary with metrics for prediction-reference pairs.
    """
    print("📊 similarity metrics (Prediction ↔ Reference):")
    for pred_idx, ref_scores in score_dict.items():
        for ref_idx, metric_values in ref_scores.items():
            # Dynamically read all metric names and values
            metric_parts = [f"{name.upper()}: {value:.3f}" for name, value in metric_values.items()]
            score_line = f"  Prediction {pred_idx} ↔ Reference {ref_idx} | " + ", ".join(metric_parts)
            print(score_line)
    print()



def print_text_segment_console_output(predictions: List[str], references: List[str], similarity_scores: Dict[int, Dict[int, Dict[str, float]]], unmatched_predictions: Set[int], unmatched_references: Set[int], matched_predictions: Set[int], matched_references: Set[int]):
    """
    Provides a clear console output for evaluating a text excerpt.

    :param predictions: List of requirements extracted by the LLM.
    :param similarity_scores: Nested dictionary with BERTScore values between predictions and references.
    :param unmatched_predictions: Indices of unmatched LLM requirements.
    :param unmatched_references: Indices of unrecognized reference requirements.
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
    Calculates the overall metrics (precision, recall, F1) for a benchmark run
    and saves them under a run ID in benchmarbenchmark_results_valuesk_run_log.

    :param run_id: Unique ID of the run (e.g., “run_1”)
    :param threshold: Currently used similarity threshold
    :param results_per_textsegment_of_all_runs: Dictionary with results per text segment
    """
    if not results_per_textsegment_of_all_runs:
        print(f"⚠️ No results found for {run_id}.")
        return

    filtered = {k: v for k, v in results_per_textsegment_of_all_runs.items() if v["run_id"] == run_id}
    df = pd.DataFrame.from_dict(filtered, orient="index")

    total_predictions = df["total_predictions"].sum()
    total_references = df["total_references"].sum()
    unmatched_predictions = df["unmatched_predictions"].sum()
    unmatched_references = df["unmatched_references"].sum()

    true_positives = total_predictions - unmatched_predictions

    # Precision: TP / (TP + FP) → corresponds to TP / total_predictions
    precision = true_positives / total_predictions if total_predictions > 0 else 0

    # Recall: TP / (TP + FN) → corresponds to TP / total_references
    recall = true_positives / total_references if total_references > 0 else 0

    # F1-Score: harmonic mean
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("\n📊 Overall result (per text segment): \n")
    print(df)

    print(f"\n📊 overall evaluation (Run: {run_id})\n") 
    print(f"📈 Precision:            {precision:.2%}")
    print(f"📈 Recall:               {recall:.2%}")
    print(f"📈 F1-Score:             {f1:.2%}")
    print(f"🎯 sum of all Predictions:   {total_predictions}")
    print(f"🎯 sum of all References:    {total_references}")



# ------------------------------------------------------- Functions for monitoring all results after completion of the runs --------------------------------------------------------


def analysis_data(results_per_textsegment_of_all_runs: dict, json_file: str):

    # Calculate result values
    compute_benchmark_results_for_each_runs(results_per_textsegment_of_all_runs)
    compute_average_results_by_threshold()
    generate_dictionary_results_per_textsegment_of_all_runs_with_textsegement_analysis_information(results_per_textsegment_of_all_runs, json_file)
    compute_benchmark_results_each_textsegment_with_subgroup_ratio()
    compute_average_results_by_threshold_with_subgroup_ratio()
    compute_benchmark_results_each_textsegment_with_subgroup_textlength()
    compute_average_results_by_threshold_with_subgroup_textlength()

    # Create charts
    plot_line_chart_precision_recall_f1score_over_threshold_per_model()
    plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category() 
    plot_boxplot_precision_recall_f1_per_model()
    plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category_from_runs()

















    
    



def compute_benchmark_results_for_each_runs(results_per_textsegment_of_all_runs: dict):
    """
    Aggregates the results of all text segments per run and calculates precision, recall, and F1.
    
    :param results_per_textsegment_of_all_runs: Dictionary with all individual results per text segment.
    :return: Dictionary benchmark_results_each_runs with aggregated values per run.
    """
    global benchmark_results_each_runs
    benchmark_results_each_runs = {}

    # Find out all run IDs
    run_ids = set(v["run_id"] for v in results_per_textsegment_of_all_runs.values())

    for run_id in run_ids:
        # Filter: only entries for this run ID
        filtered = {k: v for k, v in results_per_textsegment_of_all_runs.items() if v["run_id"] == run_id}
        df = pd.DataFrame.from_dict(filtered, orient="index")

        if df.empty:
            continue

        # Sums across all text segments of the run
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

        # Apply model and threshold from the first entry of the run
        first_entry = df.iloc[0]
        model = first_entry["model"]
        threshold = first_entry["metric_score_threshold"]

        # Save results
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

    # Console output as a table
    df_results = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")
    print("\n📊 Aggregated results of all runs:\n")
    print(df_results.round(4))



def compute_average_results_by_threshold():
    """
    Calculates the average values of all runs per model-threshold combination
    and stores them centrally in the dictionary ‘average_benchmark_results_for_each_model_threshold’.
    """
    global benchmark_results_each_runs
    global average_benchmark_results_for_each_model_threshold
    average_benchmark_results_for_each_model_threshold = {}  # Reset

    if not benchmark_results_each_runs:
        print("⚠️ No aggregated results available per run.")
        return

    # DataFrame from the run results
    df = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")

    # Grouping by model and threshold
    grouped = df.groupby(["model", "metric_score_threshold"]).mean().reset_index()

    # Fill dictionary
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

    # Console output in tabular form
    print("\n📊 Average values per model-threshold combination:\n")
    print(pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index").round(4))



def generate_dictionary_results_per_textsegment_of_all_runs_with_textsegement_analysis_information(results_per_textsegment_of_all_runs: dict, json_file: str):
    """
    Combines the benchmark results per text segment with additional analysis information
    (from EvaluateJSON.evaluate_dataset) and stores them in a global dictionary.

    :param results_per_textsegment_of_all_runs: Dictionary with results per run and text segment
    :param json_file: Path to the benchmark JSON file
    """

    global results_per_textsegment_of_all_runs_with_textsegement_analysis_information

    # Get the analysis information for all text excerpts
    text_analysis_results = EvaluateJSON.evaluate_dataset(json_file)

    # Fill new dictionary
    combined = {}
    for key, entry in results_per_textsegment_of_all_runs.items():
        text_id = str(entry["text_id"])  # text_id in Analysis-Dict is a string

        if text_id in text_analysis_results:
            # Combine benchmark results with text analysis information
            combined[key] = {**entry, **text_analysis_results[text_id]}
        else:
            # If no analysis entry exists, only transfer the benchmark data.
            combined[key] = entry

    # Save as class variable (global)
    results_per_textsegment_of_all_runs_with_textsegement_analysis_information = combined

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results_per_textsegment_of_all_runs_with_textsegement_analysis_information, orient="index")

    # Output in tabular form
    print("\n📊 Results combined with text analysis information:\n")
    print(df)



def compute_benchmark_results_each_textsegment_with_subgroup_ratio():
    """
    Aggregates the results of all text segments per run and subgroup (based on ratio_reference_to_total)
    and calculates precision, recall, and F1.

    :param results_per_textsegment_of_all_runs_with_textsegement_analysis_information: 
           Dictionary with all individual results + text analysis information per text segment.
    """
    global results_per_textsegment_of_all_runs_with_textsegement_analysis_information
    global benchmark_results_each_runs_with_subgroup_ratio
    benchmark_results_each_runs_with_subgroup_ratio = {}

    # Find out all run IDs
    run_ids = set(v["run_id"] for v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.values())

    for run_id in run_ids:
        # Filter: only entries for this run ID
        filtered = {
            k: v for k, v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.items()
            if v["run_id"] == run_id
        }
        df = pd.DataFrame.from_dict(filtered, orient="index")

        if df.empty:
            continue

        # Apply model and threshold from first entry
        first_entry = df.iloc[0]
        model = first_entry["model"]
        threshold = first_entry["metric_score_threshold"]

        # Define subgroups
        subgroup_conditions = {
            "full_relevant": df["ratio_reference_to_total"] == 1.0,
            "full_irrelevant": df["ratio_reference_to_total"] == 0.0,
            "mixed": (df["ratio_reference_to_total"] > 0.0) & (df["ratio_reference_to_total"] < 1.0),
        }

        for subgroup_name, condition in subgroup_conditions.items():
            subgroup_df = df[condition]

            if subgroup_df.empty:
                continue  # No entries in this subgroup for this run

            num_text_segments = len(subgroup_df)

            # Sums across all text segments of the subgroup
            total_predictions = subgroup_df["total_predictions"].sum()
            total_references = subgroup_df["total_references"].sum()
            unmatched_predictions = subgroup_df["unmatched_predictions"].sum()
            unmatched_references = subgroup_df["unmatched_references"].sum()

            true_positives = total_predictions - unmatched_predictions

            precision = true_positives / total_predictions if total_predictions > 0 else 0
            recall = true_positives / total_references if total_references > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            # Save results: Key = combination of run_id + subgroup
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

    # Output as table
    df_results = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_ratio, orient="index")
    print("\n📊 Aggregated results by relevance subgroups per run:\n")
    print(df_results.round(4))



def compute_average_results_by_threshold_with_subgroup_ratio():
    """
    Calculates the average values of all subgroup results per model-threshold-subgroup combination.
    Stores them centrally in the dictionary ‘average_benchmark_results_for_each_model_threshold_with_subgroup_ratio’.
    """
    global benchmark_results_each_runs_with_subgroup_ratio
    global average_benchmark_results_for_each_model_threshold_with_subgroup_ratio
    average_benchmark_results_for_each_model_threshold_with_subgroup_ratio = {}  # Reset

    if not benchmark_results_each_runs_with_subgroup_ratio:
        print("⚠️ No subgroup results available.")
        return

    # DataFrame 
    df = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_ratio, orient="index")

    # Group by model, threshold, and subgroup
    grouped = df.groupby(["model", "metric_score_threshold", "subgroup"]).mean().reset_index()

    # fill dictionary 
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

    # consol output
    print("\n📊 Durchschnittswerte je Modell-Threshold-Relevanz-Subgruppe:\n")
    print(pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold_with_subgroup_ratio, orient="index").round(4))


def compute_benchmark_results_each_textsegment_with_subgroup_textlength():
    """
    Aggregates the results of all text segments per run and subgroup (based on text length: num_words)
    and calculates precision, recall, and F1.

    Subgroups:
    - short: 0–39 words
    - medium: 40–59 words
    - long: >=60 words
    """
    global results_per_textsegment_of_all_runs_with_textsegement_analysis_information
    global benchmark_results_each_runs_with_subgroup_textlength
    benchmark_results_each_runs_with_subgroup_textlength = {}

    # Find out all run IDs
    run_ids = set(v["run_id"] for v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.values())

    for run_id in run_ids:
        # Filter: only entries for this run ID
        filtered = {
            k: v for k, v in results_per_textsegment_of_all_runs_with_textsegement_analysis_information.items()
            if v["run_id"] == run_id
        }
        df = pd.DataFrame.from_dict(filtered, orient="index")

        if df.empty:
            continue

        # Apply model and threshold from first entry
        first_entry = df.iloc[0]
        model = first_entry["model"]
        threshold = first_entry["metric_score_threshold"]

        # Define subgroups
        subgroup_conditions = {
            "short": df["num_words"] <= 39,
            "medium": (df["num_words"] >= 40) & (df["num_words"] <= 59),
            "long": df["num_words"] >= 60,
        }

        for subgroup_name, condition in subgroup_conditions.items():
            subgroup_df = df[condition]

            if subgroup_df.empty:
                continue  # No entries in this subgroup for this run

            num_text_segments = len(subgroup_df)

            # Sums across all text segments of the subgroup
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

            # Save results: Key = combination of run_id + subgroup
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

    # Output as table
    df_results = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_textlength, orient="index")
    print("\n📊 Aggregated results by text length subgroups per run:\n")
    print(df_results.round(4))


def compute_average_results_by_threshold_with_subgroup_textlength():
    """
    Calculates the average values of all subgroup results per model threshold text length subgroup.
    Stores them centrally in the dictionary ‘average_benchmark_results_for_each_model_threshold_with_subgroup_textlength’.
    """
    global benchmark_results_each_runs_with_subgroup_textlength
    global average_benchmark_results_for_each_model_threshold_with_subgroup_textlength
    average_benchmark_results_for_each_model_threshold_with_subgroup_textlength = {}  # Reset

    if not benchmark_results_each_runs_with_subgroup_textlength:
        print("⚠️ No text length subgroup results available.")
        return

    # DataFrame 
    df = pd.DataFrame.from_dict(benchmark_results_each_runs_with_subgroup_textlength, orient="index")

    # grouped by Modell, Threshold und Subgruppe
    grouped = df.groupby(["model", "metric_score_threshold", "subgroup"]).mean().reset_index()

    # fill dictionary 
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

    # consol output
    print("\n📊 Average values per model threshold text length subgroup:\n")
    print(pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold_with_subgroup_textlength,
                                 orient="index").round(4))


































def plot_line_chart_precision_recall_f1score_over_threshold_per_model():
    
    save_path = "line_chart_precision_recall_f1score_over_threshold_per_model.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil activate
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })
    

    # Prepare data
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    df = df.sort_values(by=["model", "metric_score_threshold"])
    metrics = ["precision", "recall", "f1_score"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color assignment per metric (darker shades of gray for better readability)
    cmap = plt.get_cmap("Purples")
    color_map = {
        "precision": mcolors.to_hex(cmap(0.6)),
        "recall": mcolors.to_hex(cmap(0.75)),
        "f1_score": mcolors.to_hex(cmap(0.9))
    }

    # Determine all models
    models = df["model"].unique()

    # Line charts for every metric & model
    for model in models:
        df_model = df[df["model"] == model]
        df_model = df_model[df_model["metric_score_threshold"] >= 0.75] # filters the value range from 0.75
        thresholds = df_model["metric_score_threshold"]

        for metric in metrics:
            scores = df_model[metric]
            ax.plot(
                thresholds,
                scores,
                marker="o",
                linestyle="-",
                linewidth=2,
                label=f"{metric.capitalize()}",
                color=color_map[metric]
            )

    # X-axis: Uniform and typographically consistent
    x_ticks = np.arange(0.75, 1.01, 0.05)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in x_ticks], fontsize=32)

    # Y-axis: 0–1.05, typographically consistent
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{t:.1f}" for t in np.linspace(0, 1.0, 6)], fontsize=32)

    # ax.set_yticks(np.linspace(0.5, 1.0, 3))
    # ax.set_yticks(np.linspace(0.5, 1.0, 3))
    # ax.set_yticklabels([rf"\textsf{{{t:.1f}}}" for t in np.linspace(0.5, 1.0, 3)], fontsize=18)


    # Remove axis title
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Frame lines
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Axis tick style
    ax.tick_params(left=False, bottom=False, labelsize=32)

    # Grid line Y-axis
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Legend in the usual style
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

    # Export as vector
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig("line_chart_precision_recall_f1score_over_threshold_per_model.pdf")
    # plt.show()

    print(f"\n✅ Line chart saved under: {save_path}")


def plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category():
    save_path = "benchmark_bar_chart_precision_recall_f1score_reasoning_split.png"
    global average_benchmark_results_for_each_model_threshold

    # LaTeX-Stil activate 
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # prepare data
    df = pd.DataFrame.from_dict(average_benchmark_results_for_each_model_threshold, orient="index")
    metrics = ["precision", "recall", "f1_score"]
    models = df["model"].tolist()
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    # Models sorted by parameter ranking
    reasoning_models = ["qwen3:8b", "qwen3:14b", "gpt-oss:20b", "qwen3:32b", "gpt-oss:120b", "gpt-5"]
    non_reasoning_models = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "gpt-4"]
    #models = non_reasoning_models + reasoning_models  # desired order

    models_ordered = non_reasoning_models + reasoning_models
    # Only use models that actually exist in the DataFrame
    models = [m for m in models_ordered if m in df["model"].values]

    # Prepare color gradations
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

    # Merge all colors
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
            color=all_colors.get(model, "#999999")  # Fallback-color
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

    # Axes and legend
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
        handlelength=1.25,      # Length of the color box (smaller = more compact)
        handletextpad=0.2      # Reduce spacing between color box and text
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

    print(f"\n✅ Bar chart (precision & recall, reasoning vs. non-reasoning models) saved under: {save_path}")


def plot_boxplot_precision_recall_f1_per_model():
    """
    Creates a box plot diagram of the distributions of precision, recall, and F1 score
    for each model based on all individual runs.
    """
    global benchmark_results_each_runs

    if not benchmark_results_each_runs:
        print("⚠️ No benchmark results available.")
        return

    # Set LaTeX font style (Computer Modern Roman)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # Create DataFrame
    df = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")

    # Restructure data (long format for seaborn)
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
        fliersize=2,  # Size of outliers
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

    print("\n✅ Box plot chart saved under: boxplot_precision_recall_f1_per_model.*")


def plot_benchmark_bar_chart_precision_recall_f1score_by_reasoning_category_from_runs():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import pandas as pd

    save_path = "benchmark_bar_chart_precision_recall_f1score_reasoning_split_from_runs.png"
    global benchmark_results_each_runs

    # Enable LaTeX style (Computer Modern)
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cmr"],
        "axes.unicode_minus": False
    })

    # Prepare data
    df = pd.DataFrame.from_dict(benchmark_results_each_runs, orient="index")

    # Group by model
    grouped = df.groupby("model")[["precision", "recall", "f1_score"]]
    summary = grouped.agg(["mean", "min", "max"])  # Erweiterbar: , "min", "max"
    #summary.columns = summary.columns.droplevel(1)  # Flatten multi-level index

    metrics = ["precision", "recall", "f1_score"]
    models = summary.index.tolist()
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 6))
    spacing = 0.02

    #  Models sorted by parameter ranking
    reasoning_models = ["qwen3:8b", "qwen3:14b", "gpt-oss:20b", "qwen3:32b", "gpt-oss:120b", "gpt-5"]
    non_reasoning_models = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "gpt-4"]
    models_ordered = non_reasoning_models + reasoning_models
    summary = summary.loc[[m for m in models_ordered if m in summary.index]]  # Filter & Sorting

    # Prepare color gradations
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

        # Values above the bars
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

    # Axes & Legend
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

    print(f"\n✅ Bar chart (precision, recall, F1 over runs) saved under: {save_path}")











def load_csv_and_renumber_run_ids(csv_path, chunk_size):
    """
    Reads a CSV file, saves it as a dictionary, and adjusts the run_id based on chunks.

    :param csv_path: Path to the CSV file
    :param chunk_size: Number of rows per run_id
    :return: Dictionary with newly assigned run_ids
    """
    # Import CSV (order remains unchanged)
    df = pd.read_csv(csv_path)

    if df.empty:
        print("⚠️ CSV-Datei is empty.")
        return {}

    # Calculate new run_ids: 120 lines per block
    df["run_id"] = [(i // chunk_size) + 1 for i in range(len(df))]

    # Convert to dictionary
    results_dict = df.to_dict(orient="index")

    # Control export based on the dictionary
    df_control = pd.DataFrame.from_dict(results_dict, orient="index")
    df_control.to_csv("controll_run_id_adaption.csv", index=False, encoding="utf-8")
    print("📁 Control export saved under: controll_run_id_adaption.csv")

    print(f"✅ CSV loaded and run_ids renumbered ({chunk_size} text excerpts per run).")
    return results_dict



def load_results_dict_from_csv(csv_path: str) -> dict:
    """
    Imports a CSV file and restores the dictionary results_per_textsegment_of_all_runs.
    
    :param csv_path: Path to the CSV file
    :return: Dictionary with results per text segment
    """
    df = pd.read_csv(csv_path)

    # Ensure that numeric fields are correct (optional: convert)
    df = df.convert_dtypes()

    # Convert to dictionary with int key
    results_dict = {
        idx: row._asdict() if hasattr(row, "_asdict") else row.to_dict()
        for idx, row in df.iterrows()
    }

    print(f"\n✅ CSV file successfully loaded: {csv_path}")
    return results_dict



def main():

    #csv_file = "results_per_textsegment.csv"  # Adjust the path if necessary
    csv_file = "cumulative_results_per_textsegment.csv"  # Adjust the path if necessary
    #csv_file = "cumulative_results_per_textsegment_final_version.csv"
    json_file = "BenchmarkRequirements.json"

    #results_per_textsegment_of_all_runs = load_results_dict_from_csv(csv_file)
    #results_per_textsegment_of_all_runs = load_csv_and_renumber_run_ids(csv_file, 120)
    results_per_textsegment_of_all_runs = load_csv_and_renumber_run_ids("results_per_textsegment_analysis_threshold.csv", 60)

    analysis_data(results_per_textsegment_of_all_runs, json_file)



if __name__ == "__main__":
    main()    


