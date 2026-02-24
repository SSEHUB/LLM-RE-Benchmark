import DataLoader as dl
import pandas as pd
import re

from typing import List, Dict, Set

# Ergebnis-Dictionary
text_analysis_results = {}



def count_words(text: str) -> int:
    """
    Zählt die Anzahl der Wörter in einem gegebenen Text.
    """
    if not text:
        return 0
    return len(re.findall(r'\b\w+\b', text))



def count_characters(text: str) -> int:
    """
    Zählt die Anzahl der Zeichen in einem gegebenen Text.
    """
    if not text:
        return 0
    return len(text)



def count_reference_words(requirements: Dict[str, Dict]) -> int:
    """
    Summiert die Wortanzahl aller 'gold_reference'-Texte eines Textausschnitts.
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


        # Verhältnisberechnungen
        ratio_reference_to_total = reference_words / num_words if num_words > 0 else 0
        ratio_nonref_to_total = non_reference_words / num_words if num_words > 0 else 0

        # Ergebnisse in dictionary speichern
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
    Bestimmt die Längenkategorie basierend auf der Wortanzahl des Textes.
    """
    if num_words < 10:
        return 0
    elif num_words >= 100:
        return 10
    else:
        return num_words // 10



def print_statistics():
    """Gibt grundlegende Statistiken über alle analysierten Textsegmente aus."""
    if not text_analysis_results:
        print("⚠️ Keine Analyseergebnisse vorhanden.")
        return

    df = pd.DataFrame.from_dict(text_analysis_results, orient="index")

    print("\n📊 Statistische Übersicht über alle Textsegmente:\n")

    print(f"🔢 Anzahl analysierter Textsegmente:      {len(df)}")
    print(f"📝 Gesamtanzahl Zeichen (original_text):  {df['num_chars'].sum()}")
    print(f"📝 Gesamtanzahl Wörter (original_text):   {df['num_words'].sum()}")
    print(f"🧩 Gesamtanzahl Referenzen:               {df['num_references'].sum()}")
    print(f"🧩 Gesamtanzahl Referenz-Wörter:          {df['reference_words'].sum()}")
    print(f"📉 Gesamtanzahl Nicht-Referenz-Wörter:    {df['non_reference_words'].sum()}")

    print("\n📈 Durchschnittswerte pro Textsegment:")
    print(f"➡️ Zeichen:              {df['num_chars'].mean():.2f}")
    print(f"➡️ Wörter:               {df['num_words'].mean():.2f}")
    print(f"➡️ Referenzen:           {df['num_references'].mean():.2f}")
    print(f"➡️ Referenz-Wörter:      {df['reference_words'].mean():.2f}")
    print(f"➡️ Nicht-Referenz-Wörter:{df['non_reference_words'].mean():.2f}")

    print("\n📏 Min/Max-Werte (zur Orientierung):")
    print(f"📏 Kürzester Text (Wörter):   {df['num_words'].min()} | Längster Text: {df['num_words'].max()}")
    print(f"🔢 Wenigste Referenzen:       {df['num_references'].min()} | Meiste: {df['num_references'].max()}")
    print(f"✂️ Geringste Ref-Wörter:      {df['reference_words'].min()} | Meiste: {df['reference_words'].max()}")



def print_groupwise_statistics(df: pd.DataFrame):
    """
    Gibt gruppierte Durchschnittswerte für definierte Textgruppen (alt: 1–20, 21–40, 41–60; neu: 1-40, 41-80, 81-120) aus.
    """

    # Gruppenspalte hinzufügen
    df["gruppe"] = df.index.map(assign_group)

    grouped = df.groupby("gruppe")

    # Durchschnittswerte ausgeben
    print("\n📊 Gruppierte Durchschnittswerte (Textausschnitt-Gruppen):\n")
    print(grouped[[
        "num_words",
        "num_references",
        "reference_words",
        "non_reference_words",
        "ratio_reference_to_total",
        "ratio_nonref_to_total"
    ]].mean().round(2))

    # Vollständige Kennzahlen pro Gruppe
    for group_name, group_df in grouped:
        print(f"\n📘 Gruppe {group_name} – detaillierte Kennzahlen:\n")
        print(f"🔢 Anzahl Textsegmente:         {len(group_df)}")
        print(f"📝 Gesamtanzahl Zeichen:       {group_df['num_chars'].sum()}")
        print(f"📝 Gesamtanzahl Wörter:        {group_df['num_words'].sum()}")
        print(f"🧩 Gesamtanzahl Referenzen:    {group_df['num_references'].sum()}")
        print(f"🧩 Referenz-Wörter gesamt:     {group_df['reference_words'].sum()}")
        print(f"📉 Nicht-Referenz-Wörter:      {group_df['non_reference_words'].sum()}")

        print(f"\n📈 Durchschnittswerte:")
        print(f"➡️ Zeichen:                    {group_df['num_chars'].mean():.2f}")
        print(f"➡️ Wörter:                     {group_df['num_words'].mean():.2f}")
        print(f"➡️ Referenzen:                 {group_df['num_references'].mean():.2f}")
        print(f"➡️ Referenz-Wörter:            {group_df['reference_words'].mean():.2f}")
        print(f"➡️ Nicht-Referenz-Wörter:      {group_df['non_reference_words'].mean():.2f}")

        print(f"\n📏 Min/Max-Werte:")
        print(f"📏 Kürzester Text (Wörter):    {group_df['num_words'].min()} | Längster: {group_df['num_words'].max()}")
        print(f"🔢 Wenigste Referenzen:        {group_df['num_references'].min()} | Meiste: {group_df['num_references'].max()}")
        print(f"✂️ Geringste Ref-Wörter:       {group_df['reference_words'].min()} | Meiste: {group_df['reference_words'].max()}")



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
    Gibt die Verteilung der Textausschnitte über Wortanzahl-Kategorien aus (0–10), inkl. leerer Kategorien.
    """
    print("\n📐 Kategorisierung nach Textlänge (Wortanzahl):\n")

    # Alle möglichen Kategorien von 0 bis 10
    all_categories = pd.Series(0, index=range(0, 11))

    # Tatsächliche Zählung
    actual_counts = df["length_category"].value_counts().sort_index()

    # Fehlende Kategorien ergänzen mit 0
    category_counts = all_categories.add(actual_counts, fill_value=0).astype(int)

    # Ausgabe
    for category, count in category_counts.items():
        if category < 10:
            print(f"🗂️ Kategorie {category} (Wörter {category*10}–{category*10+9}): {count} ")
        else:
            print(f"🗂️ Kategorie {category} (Wörter ≥100): {count} ")



def print_length_category_distribution_by_group(df: pd.DataFrame):
    """
    Gibt die Verteilung der Textlängen-Kategorien für jede definierte Gruppe (1–3) separat aus.
    """
    print("\n📐 Kategorisierung nach Textlänge je Gruppe (Wortanzahl):\n")

    df["gruppe"] = df.index.map(assign_group)

    for group in sorted(df["gruppe"].unique()):
        group_df = df[df["gruppe"] == group]
        print(f"\n🔹 Gruppe {group}:")

        # Kategorien: 0–10
        all_categories = pd.Series(0, index=range(0, 11))
        actual_counts = group_df["length_category"].value_counts().sort_index()
        category_counts = all_categories.add(actual_counts, fill_value=0).astype(int)

        for category, count in category_counts.items():
            if category < 10:
                print(f"🗂️ Kategorie {category} (Wörter {category*10}–{category*10+9}): {count} ")
            else:
                print(f"🗂️ Kategorie {category} (Wörter ≥100): {count} ")



def export_full_summary_to_csv(df: pd.DataFrame):
    """
    Exportiert alle Konsolenmetriken in eine strukturierte CSV-Datei.
    """

    df["gruppe"] = df.index.map(assign_group)
    output_rows = []

    # 🔢 Globale Gesamtwerte
    output_rows.append({"Kategorie": "🔢 Anzahl analysierter Textsegmente", "Wert": len(df)})
    output_rows.append({"Kategorie": "📝 Gesamtanzahl Zeichen (original_text)", "Wert": df['num_chars'].sum()})
    output_rows.append({"Kategorie": "📝 Gesamtanzahl Wörter (original_text)", "Wert": df['num_words'].sum()})
    output_rows.append({"Kategorie": "🧩 Gesamtanzahl Referenzen", "Wert": df['num_references'].sum()})
    output_rows.append({"Kategorie": "🧩 Gesamtanzahl Referenz-Wörter", "Wert": df['reference_words'].sum()})
    output_rows.append({"Kategorie": "📉 Gesamtanzahl Nicht-Referenz-Wörter", "Wert": df['non_reference_words'].sum()})

    output_rows.append({})
    output_rows.append({"Kategorie": "📈 Durchschnittswerte pro Textsegment"})
    output_rows.append({"Kategorie": "➡️ Zeichen", "Wert": round(df['num_chars'].mean(), 2)})
    output_rows.append({"Kategorie": "➡️ Wörter", "Wert": round(df['num_words'].mean(), 2)})
    output_rows.append({"Kategorie": "➡️ Referenzen", "Wert": round(df['num_references'].mean(), 2)})
    output_rows.append({"Kategorie": "➡️ Referenz-Wörter", "Wert": round(df['reference_words'].mean(), 2)})
    output_rows.append({"Kategorie": "➡️ Nicht-Referenz-Wörter", "Wert": round(df['non_reference_words'].mean(), 2)})

    output_rows.append({})
    output_rows.append({"Kategorie": "📏 Min/Max-Werte"})
    output_rows.append({"Kategorie": "📏 Kürzester Text (Wörter)", "Wert": df['num_words'].min()})
    output_rows.append({"Kategorie": "📏 Längster Text (Wörter)", "Wert": df['num_words'].max()})
    output_rows.append({"Kategorie": "🔢 Wenigste Referenzen", "Wert": df['num_references'].min()})
    output_rows.append({"Kategorie": "🔢 Meiste Referenzen", "Wert": df['num_references'].max()})
    output_rows.append({"Kategorie": "✂️ Geringste Ref-Wörter", "Wert": df['reference_words'].min()})
    output_rows.append({"Kategorie": "✂️ Meiste Ref-Wörter", "Wert": df['reference_words'].max()})

    # 📐 Kategorisierung nach Textlänge
    output_rows.append({})
    output_rows.append({"Kategorie": "📐 Kategorisierung nach Textlänge (Wortanzahl)"})
    category_counts = df['length_category'].value_counts().reindex(range(0, 11), fill_value=0).sort_index()
    for cat, count in category_counts.items():
        cat_range = f"{cat*10}–{cat*10+9}" if cat < 10 else "≥100"
        output_rows.append({"Kategorie": f"🗂️ Kategorie {cat} (Wörter {cat_range})", "Wert": count})

    # 📊 Gruppierte Durchschnittswerte
    grouped = df.groupby("gruppe")
    output_rows.append({})
    output_rows.append({"Kategorie": "📊 Gruppierte Durchschnittswerte"})

    for group, gdf in grouped:
        output_rows.append({})
        output_rows.append({"Kategorie": f"📘 Gruppe {group} – Gesamtsummen"})
        output_rows.append({"Kategorie": "Anzahl Textsegmente", "Wert": len(gdf)})
        output_rows.append({"Kategorie": "Zeichen gesamt", "Wert": gdf['num_chars'].sum()})
        output_rows.append({"Kategorie": "Wörter gesamt", "Wert": gdf['num_words'].sum()})
        output_rows.append({"Kategorie": "Referenzen gesamt", "Wert": gdf['num_references'].sum()})
        output_rows.append({"Kategorie": "Referenz-Wörter gesamt", "Wert": gdf['reference_words'].sum()})
        output_rows.append({"Kategorie": "Nicht-Referenz-Wörter", "Wert": gdf['non_reference_words'].sum()})

        output_rows.append({"Kategorie": f"📘 Gruppe {group} – Durchschnittswerte"})
        output_rows.append({"Kategorie": "Ø Zeichen", "Wert": round(gdf['num_chars'].mean(), 2)})
        output_rows.append({"Kategorie": "Ø Wörter", "Wert": round(gdf['num_words'].mean(), 2)})
        output_rows.append({"Kategorie": "Ø Referenzen", "Wert": round(gdf['num_references'].mean(), 2)})
        output_rows.append({"Kategorie": "Ø Ref-Wörter", "Wert": round(gdf['reference_words'].mean(), 2)})
        output_rows.append({"Kategorie": "Ø NonRef-Wörter", "Wert": round(gdf['non_reference_words'].mean(), 2)})

        output_rows.append({"Kategorie": "Ø Ref/Total", "Wert": f"{gdf['ratio_reference_to_total'].mean():.2%}"})
        output_rows.append({"Kategorie": "Ø NonRef/Total", "Wert": f"{gdf['ratio_nonref_to_total'].mean():.2%}"})

        output_rows.append({"Kategorie": f"📘 Gruppe {group} – Min/Max"})
        output_rows.append({"Kategorie": "Kürzester Text (Wörter)", "Wert": gdf['num_words'].min()})
        output_rows.append({"Kategorie": "Längster Text (Wörter)", "Wert": gdf['num_words'].max()})
        output_rows.append({"Kategorie": "Min Referenzen", "Wert": gdf['num_references'].min()})
        output_rows.append({"Kategorie": "Max Referenzen", "Wert": gdf['num_references'].max()})
        output_rows.append({"Kategorie": "Min Ref-Wörter", "Wert": gdf['reference_words'].min()})
        output_rows.append({"Kategorie": "Max Ref-Wörter", "Wert": gdf['reference_words'].max()})

        # Kategorie-Auswertung pro Gruppe
        group_categories = gdf['length_category'].value_counts().reindex(range(0, 11), fill_value=0).sort_index()
        for cat, count in group_categories.items():
            cat_range = f"{cat*10}–{cat*10+9}" if cat < 10 else "≥100"
            output_rows.append({"Kategorie": f"🗂️ Gruppe {group} – Kategorie {cat} (Wörter {cat_range})", "Wert": count})

    # Export
    summary_df = pd.DataFrame(output_rows)
    summary_df.to_csv("text_analysis_summary.csv", index=False)
    print("\n📄 Ausführlicher Bericht wurde in 'text_analysis_summary.csv' gespeichert.")



def main():
    """
    Main function to start the evaluation process.
    """
    json_file = "BenchmarkRequirements.json"  # Adjust the path if necessary
    evaluate_dataset(json_file)

    # Ausgabe als DataFrame
    df = pd.DataFrame.from_dict(text_analysis_results, orient="index")
    print("\n📊 Analyse abgeschlossen. Übersicht:\n")
    print(df)

    # Optional: als CSV speichern
    df.to_csv("text_analysis_results.csv", index_label="text_id")
    print("\n💾 Ergebnisse wurden in 'text_analysis_results.csv' gespeichert.")

    # Statistikfunktion
    print_statistics()
    print_length_category_distribution(df)

    # Gruppierte Auswertung
    print_groupwise_statistics(df)

    # Gruppierte Kategorisierung
    print_length_category_distribution_by_group(df)

    # Zusammenfassung in zweiter CSV-Datei
    export_full_summary_to_csv(df)



if __name__ == "__main__":
    main()    