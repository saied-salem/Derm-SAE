import os
import csv
import json

print("Starting vocabulary preparation...")

# --- File paths ---
csv_file = 'concept.csv'
json_file = 'selected_concepts.json'
output_file = 'master_vocabulary.txt'

# --- Check if source files exist ---
if not os.path.exists(csv_file) or not os.path.exists(json_file):
    print(f"Error: Missing source files.")
    print(f"Please make sure '{csv_file}' and '{json_file}' are in this directory.")
else:
    all_concepts = set()

    # --- Load concepts from concept.csv ---
    try:
        print(f"Loading concepts from {csv_file}...")
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip the header row
            header = next(reader)
            
            count = 0
            for row in reader:
                if len(row) > 2: # Ensure row has the concept column
                    # Get the string from the 3rd column (index 2)
                    concept_string = row[2].strip().lower()
                    if not concept_string:
                        continue
                        
                    # Split concepts that are comma-separated
                    # e.g., "macule, melanotic" -> ["macule", "melanotic"]
                    individual_concepts = concept_string.split(',')
                    
                    for concept in individual_concepts:
                        cleaned_concept = concept.strip()
                        if cleaned_concept:
                            all_concepts.add(cleaned_concept)
                            count += 1
            print(f"Loaded and parsed {count} concept entries from {csv_file}.")
            
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

    # --- Load concepts from selected_concepts.json ---
    try:
        print(f"Loading concepts from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            count = 0
            # The JSON is a dictionary where keys are diseases
            # and values are lists of concept strings.
            # We only want the values (the lists).
            for concept_list in data.values():
                for concept in concept_list:
                    cleaned_concept = str(concept).strip().lower()
                    if cleaned_concept:
                        all_concepts.add(cleaned_concept)
                        count += 1
            print(f"Loaded {count} concepts from {json_file}.")

    except Exception as e:
        print(f"Error reading {json_file}: {e}")

    # --- De-duplicate and sort ---
    if all_concepts:
        sorted_concepts = sorted(list(all_concepts))

        # --- Save to master file ---
        with open(output_file, 'w') as f:
            for concept in sorted_concepts:
                f.write(f"{concept}\n")
        
        print(f"\nSuccessfully created '{output_file}' with {len(sorted_concepts)} unique concepts.")
    else:
        print("\nNo concepts were loaded. Output file was not created.")