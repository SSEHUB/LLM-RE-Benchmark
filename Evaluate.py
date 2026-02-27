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

    # Calculation of all metric values (Rouge) at once --> Output: Matrix between all predictions and references (if no reference is available, the note “This text does not ...” is introduced as a reference)
    similarity_scores = calculate_all_metrics_for_text_segment(list_of_predictions, extract_reference_with_placeholder(text_segment["requirements"]))

    # Calculation of results for a text segment
    text_segment_results = determine_text_segment_matchings(list_of_predictions, extract_reference_with_placeholder(text_segment["requirements"]), similarity_scores, metric_score_threshold)

    # Store results
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
    Returns 0 if the list contains only one prediction and it is empty,
    or contains the special note text. Otherwise, the number of predictions is returned.
    """
    placeholder = "This text does not contain any functional requirement information"

    if len(predictions) == 1:
        cleaned = predictions[0].strip()
        if cleaned == "" or cleaned == placeholder:
            return 0

    return len(predictions)



def count_valid_references(reference_groups: List[List[str]]) -> int:
    """
    Returns 0 if there is only one reference group and it is empty,
    or contains only the note text. Otherwise, the number of reference groups is returned.
    """
    placeholder = "This text does not contain any functional requirement information."

    if len(reference_groups) == 1:
        single_ref = reference_groups[0]
        # Clean up empty entries (“” or None)
        cleaned = [r.strip() for r in single_ref if r and r.strip()]
        if not cleaned or (len(cleaned) == 1 and cleaned[0] == placeholder):
            return 0

    return len(reference_groups)



def determine_text_segment_matchings(predictions: List[str], references: List[List[str]], similarity_scores: dict, metric_score_threshold: float) -> dict:
    """
    Evaluates the performance of an LLM in extracting requirements from a text excerpt.

    :param predictions: List of requirements extracted by the LLM.
    :param references: List of lists with reference requirements (gold standard + alternatives).
    :return: Score for this text excerpt.
    """
    unmatched_references = set(range(len(references)))  # Indexes of unrecognized references
    unmatched_predictions = set(range(len(predictions)))  # Indexes of unassigned predictions
    matched_references = []
    matched_predictions = []

    # Matching each prediction with the best matching reference
    for pred_idx in similarity_scores:
        best_score = 0
        best_ref_idx = None

        for ref_idx in similarity_scores[pred_idx]:
            current_score = similarity_scores[pred_idx][ref_idx]["rouge1"]  # Here you select the score, e.g. f1.

            if current_score > best_score:
                best_score = current_score
                best_ref_idx = ref_idx

        # If a suitable reference is found (e.g., BERTScore ≥ 0.9), mark it as recognized.
        if best_ref_idx is not None and best_score >= metric_score_threshold:
            unmatched_references.discard(best_ref_idx)
            matched_references.append(best_ref_idx)
            unmatched_predictions.discard(pred_idx)
            matched_predictions.append(pred_idx)

    mr.print_text_segment_console_output(predictions, references, similarity_scores, unmatched_predictions, unmatched_references, matched_predictions, matched_references)

    # Special case: Placeholder reference (“The System does not ...”) should not count in the evaluation and is therefore removed. 
    if len(unmatched_references) == 1 and count_valid_references(references) == 0: # if a placeholder is present, then this is the sole reference
        unmatched_references.clear()

    # Special case: Placeholder prediction (“The System does not ...”) should not count in the evaluation and is therefore removed if present.
    # Since the placeholder is LLM-generated, alternatives are also removed if they match a placeholder reference.
    total_predictions = count_valid_predictions(predictions) # Placeholders do not count and are deducted if they were generated by the LLM with the exact defined wording. 
    if (total_predictions == 1 and len(matched_predictions) == 1 and count_valid_references(references) == 0): # LLM-generated placeholder could be matched, but does not correspond to the exact wording and must therefore be excluded in this step.
        total_predictions = 0

    return {
        "unmatched_predictions": len(unmatched_predictions),
        "unmatched_references": len(unmatched_references),
        "total_predictions": total_predictions  # returns the number of all predictions, minus (matched) placeholders
    }



def calculate_all_metrics_for_text_segment(predictions: List[str], references: List[List[str]]) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Calculates BLEU, ROUGE-1, and METEOR for each prediction relative to each reference request.

    :param predictions: List of requests generated by the LLM.
    :param references: List of lists containing reference requests (gold + alternatives).
    :return: Nested dictionary with scores for each prediction-reference pairing.
    """
    results_dict = {}

    for pred_idx, prediction in enumerate(predictions):
        results_dict[pred_idx] = {}

        for ref_idx, reference_list in enumerate(references):
            # calculation
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
    Returns reference requirements. If none exist, a placeholder requirement is inserted.

    :param requirements: Dictionary of requirements for a text segment.
    :return: List of lists with reference requirements (including placeholders if necessary).
    """
    formatted_references = extract_reference(requirements)

    if not any(formatted_references):  # checks whether all inner lists are empty
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
    Reads the CSV files ‘cumulative_results_per_textsegment.csv’ and ‘results_per_textsegment.csv’,
    merges them line by line, and saves the result as a new ‘cumulative_results_per_textsegment.csv’.
    Existing data in the cumulative file remains unchanged unless
    duplicates (identical lines) occur.
    """
    cumulative_path = "cumulative_results_per_textsegment.csv"
    current_run_path = "results_per_textsegment.csv"

    # Check whether both files exist
    if not os.path.exists(current_run_path):
        print("❌ Error: 'results_per_textsegment.csv' was not found.")
        return

    # Download both files
    df_new = pd.read_csv(current_run_path)

    if os.path.exists(cumulative_path):
        df_cumulative = pd.read_csv(cumulative_path)
        # Merge (append) – Remove duplicates, if any
        combined_df = pd.concat([df_cumulative, df_new], ignore_index=True)
        print("🔁 Existing cumulative file has been supplemented.")
    else:
        combined_df = df_new
        print("🆕 New cumulative file has been created.")

    # Save result
    combined_df.to_csv(cumulative_path, index=False, encoding="utf-8")
    print(f"✅ Merged CSV saved under: {cumulative_path}")



def export_append_results_per_textsegment_to_csv(results_per_textsegment_of_all_runs: dict):
    """
    Adds the dictionary results_per_textsegment_of_all_runs to the existing CSV file
    or creates it if it does not exist.

    :param results_per_textsegment_of_all_runs: Dictionary with the results of all text segments
    """
    save_path = "interim_results_per_textsegment.csv"

    if not results_per_textsegment_of_all_runs:
        print("⚠️ No content available for export.")
        return

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(results_per_textsegment_of_all_runs, orient="index")

    # CSV export
    df.to_csv(save_path, index=False, encoding="utf-8")

    print(f"\n✅ CSV file with the results saved under: {save_path}")



def export_results_per_textsegment_to_csv(results_per_textsegment_of_all_runs: dict):
    """
    Exports the dictionary results_per_textsegment_of_all_runs to a CSV file.

    :param results_per_textsegment_of_all_runs: Dictionary with the results of all text segments
    :param save_path: Storage path for the CSV file
    """

    save_path = "results_per_textsegment.csv"

    if not results_per_textsegment_of_all_runs:
        print("⚠️ No content available for export.")
        return

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(results_per_textsegment_of_all_runs, orient="index")

    # CSV export
    df.to_csv(save_path, index=False, encoding="utf-8")

    print(f"\n✅ CSV file with the results saved under: {save_path}")



def run_multiple_evaluations(json_file: str, thresholds: List[float], models: List[str], runs_per_model: int) :
    """
    Performs the evaluation of the data set multiple times.

    :param json_file: Path to the JSON file containing the benchmark data.
    :param runs: Number of repetitions.
    """
    global results_per_textsegment_of_all_runs

    run_id = 1

    for model in models:
        print(f"\n🤖 Start runs for model: {model}\n")

        for threshold in thresholds:
            print(f"\n🎯 Starte runs for threshold {threshold} (Model: {model}, {runs_per_model} repetitions)\n")

            for m_run in range(runs_per_model):  # Repetitions per model
                print(f"\n🚀 Starting evaluation run {run_id} (Model: {model}, Threshold: {threshold}, model-run {m_run+1})")

                # Evaluate data set
                evaluate_dataset(json_file, run_id=run_id, metric_score_threshold=threshold, model=model)

                # Summarize results
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
        models=["gpt-5"],
        runs_per_model=5
    )
    


if __name__ == "__main__":
    main()