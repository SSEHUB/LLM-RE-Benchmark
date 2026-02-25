import DataLoader as dl
import pandas as pd
import re

from typing import List, Dict, Set

# result-dictionary
text_analysis_results = {}



def count_words(text: str) -> int:
    """
    Counts the number of words in a given text.
    """
    if not text:
        return 0
    return len(re.findall(r'\b\w+\b', text))



def count_characters(text: str) -> int:
    """
    Counts the number of characters in a given text.
    """
    if not text:
        return 0
    return len(text)



def count_reference_words(requirements: Dict[str, Dict]) -> int:
    """
    Adds up the word count of all gold_reference texts in a text excerpt.
    """
    return sum(count_words(req_data["gold_reference"]) for req_data in requirements.values() if req_data["gold_reference"])



def evaluate_dataset(json_file: str):

    global text_analysis_results
    data = dl.load_json_to_dict(json_file)

    if not data:
        print("No data found in JSON file.")
        return

    #print(f"\n✅ Loaded {len(data)} text segments. Starting analysis...\n")

    for text_id, entry in data.items():
        original_text = entry["original_text"]
        requirements = entry["requirements"]

        num_chars = count_characters(original_text)
        num_words = count_words(original_text)
        num_references = len(requirements)
        reference_words = count_reference_words(requirements)
        non_reference_words = num_words - reference_words
        length_category = determine_length_category(num_words)


        # ratio calculations
        ratio_reference_to_total = reference_words / num_words if num_words > 0 else 0
        ratio_nonref_to_total = non_reference_words / num_words if num_words > 0 else 0

        # Save results in dictionary
        text_analysis_results[text_id] = {
            "source": entry["source"],
            "pages": entry["pages"],
            "num_chars": num_chars,
            "num_words": num_words,
            "num_references": num_references,
            "reference_words": reference_words,
            "non_reference_words": non_reference_words,
            "ratio_reference_to_total": round(ratio_reference_to_total, 2),
            "ratio_nonref_to_total": round(ratio_nonref_to_total, 2),
            "length_category": length_category
        }

    return text_analysis_results



def determine_length_category(num_words: int) -> int:
    """
    Determines the length category based on the number of words in the text.
    """
    if num_words < 10:
        return 0
    elif num_words >= 100:
        return 10
    else:
        return num_words // 10



def print_statistics():
    """Provides basic statistics on all analyzed text segments."""
    if not text_analysis_results:
        print("⚠️ No analysis results available.")
        return

    df = pd.DataFrame.from_dict(text_analysis_results, orient="index")

    print("\n📊 Statistical overview of all text segments:\n")

    print(f"🔢 Number of text segments analyzed:           {len(df)}")
    print(f"📝 Total number of characters (original_text): {df['num_chars'].sum()}")
    print(f"📝 Total number of words (original_text):      {df['num_words'].sum()}")
    print(f"🧩 Total number of references:                 {df['num_references'].sum()}")
    print(f"🧩 Total number of reference words:            {df['reference_words'].sum()}")
    print(f"📉 Total number of non-reference words:        {df['non_reference_words'].sum()}")

    print("\n📈 Average values per text segment:")
    print(f"➡️ Characters:           {df['num_chars'].mean():.2f}")
    print(f"➡️ Words:                {df['num_words'].mean():.2f}")
    print(f"➡️ References:           {df['num_references'].mean():.2f}")
    print(f"➡️ Reference words:      {df['reference_words'].mean():.2f}")
    print(f"➡️ Non-reference words:  {df['non_reference_words'].mean():.2f}")

    print("\n📏 Min/Max values:")
    print(f"📏 Shortest text (words):   {df['num_words'].min()} | Longest text: {df['num_words'].max()}")
    print(f"🔢 Fewest reference:       {df['num_references'].min()} | Most: {df['num_references'].max()}")
    print(f"✂️ Fewest ref words:      {df['reference_words'].min()} | Most: {df['reference_words'].max()}")



def print_groupwise_statistics(df: pd.DataFrame):
    """
    Returns grouped average values for defined text groups (old: 1 to 20, 21 to 40, 41 to 60; new: 1 to 40, 41 to 80, 81 to 120).
    """

    # Add group column
    df["group"] = df.index.map(assign_group)

    grouped = df.groupby("group")

    # Output average values
    print("\n📊 Grouped average values (text excerpt groups):\n")
    print(grouped[[
        "num_words",
        "num_references",
        "reference_words",
        "non_reference_words",
        "ratio_reference_to_total",
        "ratio_nonref_to_total"
    ]].mean().round(2))

    # Complete key figures per group
    for group_name, group_df in grouped:
        print(f"\n📘 Group {group_name} – Detailed key figures:\n")
        print(f"🔢 Number of text segments:         {len(group_df)}")
        print(f"📝 Total number of characters:     {group_df['num_chars'].sum()}")
        print(f"📝 Total number of words:          {group_df['num_words'].sum()}")
        print(f"🧩 Total number of references:     {group_df['num_references'].sum()}")
        print(f"🧩 Total reference words:          {group_df['reference_words'].sum()}")
        print(f"📉 Non-reference words:            {group_df['non_reference_words'].sum()}")

        print(f"\n📈 Average values:")
        print(f"➡️ Characters:                      {group_df['num_chars'].mean():.2f}")
        print(f"➡️ Words:                           {group_df['num_words'].mean():.2f}")
        print(f"➡️ References:                      {group_df['num_references'].mean():.2f}")
        print(f"➡️ Reference words:                 {group_df['reference_words'].mean():.2f}")
        print(f"➡️ Non-reference words:             {group_df['non_reference_words'].mean():.2f}")

        print(f"\n📏 Min/Max values:")
        print(f"📏 Shortest text (words):         {group_df['num_words'].min()} | Longest: {group_df['num_words'].max()}")
        print(f"🔢 Fewest references:             {group_df['num_references'].min()} | Most: {group_df['num_references'].max()}")
        print(f"✂️ Fewest ref words:               {group_df['reference_words'].min()} | Most: {group_df['reference_words'].max()}")



def assign_group(text_id: str) -> str:
    number = int(text_id)
    if 1 <= number <= 40:
        return 1
    elif 41 <= number <= 80:
        return 2
    elif 81 <= number <= 120:
        return 3
    else:
        return 4



def print_length_category_distribution(df: pd.DataFrame):
    """
    Outputs the distribution of text excerpts across word count categories (0 to 10), including empty categories.
    """
    print("\n📐 Categorization by text length (number of words):\n")

    # All possible categories from 0 to 10
    all_categories = pd.Series(0, index=range(0, 11))

    # Actual count
    actual_counts = df["length_category"].value_counts().sort_index()

    # Add missing categories with 0
    category_counts = all_categories.add(actual_counts, fill_value=0).astype(int)

    # Output
    for category, count in category_counts.items():
        if category < 10:
            print(f"🗂️ Category {category} (words {category*10} to {category*10+9}): {count} ")
        else:
            print(f"🗂️ Category {category} (words ≥100): {count} ")



def print_length_category_distribution_by_group(df: pd.DataFrame):
    """
    Outputs the distribution of text length categories separately for each defined group (1 to 3).
    """
    print("\n📐 Categorization by text length per group (number of words):\n")

    df["group"] = df.index.map(assign_group)

    for group in sorted(df["group"].unique()):
        group_df = df[df["group"] == group]
        print(f"\n🔹 group {group}:")

        # Category: 0–10
        all_categories = pd.Series(0, index=range(0, 11))
        actual_counts = group_df["length_category"].value_counts().sort_index()
        category_counts = all_categories.add(actual_counts, fill_value=0).astype(int)

        for category, count in category_counts.items():
            if category < 10:
                print(f"🗂️ Category {category} (words {category*10} to {category*10+9}): {count} ")
            else:
                print(f"🗂️ Category {category} (words ≥100): {count} ")



def export_full_summary_to_csv(df: pd.DataFrame):
    """
    Exports all console metrics to a structured CSV file.
    """

    df["group"] = df.index.map(assign_group)
    output_rows = []

    # 🔢 Global total values
    output_rows.append({"Category": "🔢 Number of text segments analyzed", "Value": len(df)})
    output_rows.append({"Category": "📝 Total number of words (original_text)", "Value": df['num_chars'].sum()})
    output_rows.append({"Category": "📝 Total number of words (original_text)", "Value": df['num_words'].sum()})
    output_rows.append({"Category": "🧩 Total number of references", "Value": df['num_references'].sum()})
    output_rows.append({"Category": "🧩 Total number of reference words", "Value": df['reference_words'].sum()})
    output_rows.append({"Category": "📉 Total number of non-reference words", "Value": df['non_reference_words'].sum()})

    output_rows.append({})
    output_rows.append({"Category": "📈 Average values per text segment"})
    output_rows.append({"Category": "➡️ Characters", "Value": round(df['num_chars'].mean(), 2)})
    output_rows.append({"Category": "➡️ Words", "Value": round(df['num_words'].mean(), 2)})
    output_rows.append({"Category": "➡️ References", "Value": round(df['num_references'].mean(), 2)})
    output_rows.append({"Category": "➡️ Reference words", "Value": round(df['reference_words'].mean(), 2)})
    output_rows.append({"Category": "➡️ Non-reference words", "Value": round(df['non_reference_words'].mean(), 2)})

    output_rows.append({})
    output_rows.append({"Category": "📏 Min/Max values"})
    output_rows.append({"Category": "📏 Shortest text (words)", "Value": df['num_words'].min()})
    output_rows.append({"Category": "📏 Longest text (words)", "Value": df['num_words'].max()})
    output_rows.append({"Category": "🔢 Fewest References", "Value": df['num_references'].min()})
    output_rows.append({"Category": "🔢 Most References", "Value": df['num_references'].max()})
    output_rows.append({"Category": "✂️ Fewest ref words", "Value": df['reference_words'].min()})
    output_rows.append({"Category": "✂️ Most ref words", "Value": df['reference_words'].max()})

    # 📐 Categorization by text length
    output_rows.append({})
    output_rows.append({"Category": "📐 Categorization by text length (number of words)"})
    category_counts = df['length_category'].value_counts().reindex(range(0, 11), fill_value=0).sort_index()
    for cat, count in category_counts.items():
        cat_range = f"{cat*10}–{cat*10+9}" if cat < 10 else "≥100"
        output_rows.append({"Category": f"🗂️ Category {cat} (Words {cat_range})", "Value": count})

    # 📊 Grouped average values
    grouped = df.groupby("group")
    output_rows.append({})
    output_rows.append({"Category": "📊 Grouped average values"})

    for group, gdf in grouped:
        output_rows.append({})
        output_rows.append({"Category": f"📘 Group {group} – Totals"})
        output_rows.append({"Category": "Number of text segments", "Value": len(gdf)})
        output_rows.append({"Category": "Total characters", "Value": gdf['num_chars'].sum()})
        output_rows.append({"Category": "Total words", "Value": gdf['num_words'].sum()})
        output_rows.append({"Category": "Total references", "Value": gdf['num_references'].sum()})
        output_rows.append({"Category": "Total reference words", "Value": gdf['reference_words'].sum()})
        output_rows.append({"Category": "Non-reference words", "Value": gdf['non_reference_words'].sum()})

        output_rows.append({"Category": f"📘 Group {group} – Average values"})
        output_rows.append({"Category": "Ø Characters", "Value": round(gdf['num_chars'].mean(), 2)})
        output_rows.append({"Category": "Ø Words", "Value": round(gdf['num_words'].mean(), 2)})
        output_rows.append({"Category": "Ø References", "Value": round(gdf['num_references'].mean(), 2)})
        output_rows.append({"Category": "Ø Ref words", "Value": round(gdf['reference_words'].mean(), 2)})
        output_rows.append({"Category": "Ø NonRef words", "Value": round(gdf['non_reference_words'].mean(), 2)})

        output_rows.append({"Category": "Ø Ref/Total", "Value": f"{gdf['ratio_reference_to_total'].mean():.2%}"})
        output_rows.append({"Category": "Ø NonRef/Total", "Value": f"{gdf['ratio_nonref_to_total'].mean():.2%}"})

        output_rows.append({"Category": f"📘 Group {group} – Min/Max"})
        output_rows.append({"Category": "Shortest text (words)", "Value": gdf['num_words'].min()})
        output_rows.append({"Category": "Longest text (words)", "Value": gdf['num_words'].max()})
        output_rows.append({"Category": "Min References", "Value": gdf['num_references'].min()})
        output_rows.append({"Category": "Max References", "Value": gdf['num_references'].max()})
        output_rows.append({"Category": "Min Ref Words", "Value": gdf['reference_words'].min()})
        output_rows.append({"Category": "Max Ref Words", "Value": gdf['reference_words'].max()})

        # Category evaluation per group
        group_categories = gdf['length_category'].value_counts().reindex(range(0, 11), fill_value=0).sort_index()
        for cat, count in group_categories.items():
            cat_range = f"{cat*10}–{cat*10+9}" if cat < 10 else "≥100"
            output_rows.append({"Category": f"🗂️ Group {group} – Category {cat} (Words {cat_range})", "Value": count})

    # Export
    summary_df = pd.DataFrame(output_rows)
    summary_df.to_csv("text_analysis_summary.csv", index=False)
    print("\n📄 Detailed report saved in 'text_analysis_summary.csv'")



def main():
    """
    Main function to start the evaluation process.
    """
    json_file = "BenchmarkRequirements.json"  # Adjust the path if necessary
    evaluate_dataset(json_file)

    # Output as DataFrame
    df = pd.DataFrame.from_dict(text_analysis_results, orient="index")
    print("\n📊 Analysis complete. Overview:\n")
    print(df)

    # Optional: Save as CSV
    df.to_csv("text_analysis_results.csv", index_label="text_id")
    print("\n💾 Results were saved in text_analysis_results.csv.")

    # statistical function
    print_statistics()
    print_length_category_distribution(df)

    # grouped evaluation
    print_groupwise_statistics(df)

    # Grouped categorization
    print_length_category_distribution_by_group(df)

    # Summary in second CSV file
    export_full_summary_to_csv(df)



if __name__ == "__main__":
    main()    