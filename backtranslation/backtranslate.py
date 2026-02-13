import json
from deep_translator import GoogleTranslator
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
code_folder = current_dir.parent / "data" / "code"
sys.path.append(str(code_folder))

try:
    from config_raw_data import LANGUAGES
    print(f"Success! Imported Languages: {LANGUAGES}")
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

languages = LANGUAGES

for language in languages:
    translator = GoogleTranslator(source=language, target='en')
    perturbations = ["alteration", "expansion_impact", "expansion_noimpact", "intensifier", "omission", "spelling", "synonym", "word_order"]
 
    for perturbation in perturbations:
        input_file = f"../contratico/en-{language}/{perturbation}.jsonl"
        # output_file = f"en-{language}/bt-{perturbation}.jsonl"
        output_file = Path(f"en-{language}/bt-{perturbation}.jsonl")

        updated_jsonl = []
        # --- LIMITER START ---
        count = 0 
        limit = 10
        # ---------------------

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= limit: # Check if we reached 10
                    break
                
                data = json.loads(line.strip())
                pert_key = f"{language}"
                
                if pert_key in data:
                    print(f"[{count+1}/10] Perturbed translation: ", data[pert_key])
                    try:
                        translated_text = translator.translate(data[pert_key])
                        print("Backtranslation: ", translated_text)
                        data[f"bt_{pert_key}"] = translated_text 
                        count += 1 # Increment only on successful attempts or iterations
                    except Exception as e:
                        print(f"Translation failed: {e}")
                        data[f"bt_{pert_key}"] = ""
                        count += 1
                    
                    updated_jsonl.append(data)
                    
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in updated_jsonl:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')