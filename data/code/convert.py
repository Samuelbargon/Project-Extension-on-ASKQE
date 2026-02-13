import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

def tsv_to_jsonl(input_path, output_path):
    """
    Reads a TSV file and writes it to a JSONL file.
    Uses the 'targetLang' column dynamically for the key.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as tsv_file, \
             open(output_path, 'w', encoding='utf-8') as jsonl_file:
            
            reader = csv.DictReader(tsv_file, delimiter='\t')
            
            count = 0
            
            for row in reader:
                target_language_key = row.get('targetLang')
                
                if not target_language_key:
                    continue

                entry = {
                    "id": row['stringID'],
                    "en": row['sourceString'],
                    target_language_key: row['targetString']
                }
                
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1

        print(f"-> Converted: {input_path.name} ({count} lines)")

    except Exception as e:
        print(f"X Error converting {input_path.name}: {e}")

def run_batch_conversion():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for TSV files in: {RAW_DIR}")
    
    tsv_files = list(RAW_DIR.glob("*.tsv"))
    
    if not tsv_files:
        print("No .tsv files found to convert.")
        return

    for input_file in tsv_files:
        clean_name = input_file.stem.replace("dev.", "") 
        output_filename = f"{clean_name}.jsonl"
        output_file = PROCESSED_DIR / output_filename

        if output_file.exists():
            print(f"-> Skipped: {output_filename} (Already exists)")
            continue
            
        tsv_to_jsonl(input_file, output_file)

    print("\nBatch conversion process finished.")

if __name__ == "__main__":
    run_batch_conversion()