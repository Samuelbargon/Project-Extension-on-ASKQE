import json
import os
from comet import download_model, load_from_checkpoint
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
code_folder = current_dir.parent.parent / "data" / "code"
sys.path.append(str(code_folder))

try:
    from config_raw_data import LANGUAGES
    print(f"Success! Imported Languages: {LANGUAGES}")
except ImportError as e:
    print(f"Error importing: {e}")
    print(f"Debug: Tried to look in {code_folder}")
    sys.exit(1)

languages = LANGUAGES

model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)


# languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]

for language in languages:
    for perturbation in perturbations:
        jsonl_file = f"en-{language}/{perturbation}.jsonl"
        print(f"\nProcessing File: {jsonl_file}")

        if not os.path.exists(jsonl_file):
            print(f"File not found: {jsonl_file}")
            continue

        data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if "en" in item and f"pert_{language}" in item:
                    data.append({"src": item["en"], "mt": item[f"pert_{language}"], "ref": None})

        if not data:
            print(f"No valid data in {jsonl_file}")
            continue

        scores = model.predict(data)
        total_score = sum(scores["scores"])
        num_entries = len(scores["scores"])
        average_score = total_score / num_entries if num_entries > 0 else 0

        print(f"Average COMET Score: {average_score:.4f}")
