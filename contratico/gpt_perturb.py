import openai
import json
from prompt import prompts
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
    print(f"Debug: Tried to look in {code_folder}")
    sys.exit(1)

languages = LANGUAGES

OPENAI_API_KEY = ""

LANGUAGE_MAP = {
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "tl": "Tagalog",
    "zh": "Chinese"
}

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def call_chatgpt_turbo(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# languages = ["es", "fr", "hi", "tl", "zh"]
perturbations = ["synonym", "word_order", "spelling", "expansion_noimpact", 
                 "intensifier", "expansion_impact", "omission", "alteration"]


for language in languages:
    for perturbation in perturbations:
        print("Perturbation: ", perturbation)
        
        input_file = f"../data/processed/en-{language}.jsonl"
        output_file_path = Path(f"en-{language}/{perturbation}.jsonl")

        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as file, \
             open(str(output_file_path), "w", encoding="utf-8") as out_file:
            
            for line in file:
                data = json.loads(line)
                if f"{language}" in data:
                    target_lang = LANGUAGE_MAP.get(language, language)
                    sentence = data[f"{language}"]
                    prompt = prompts[f"{perturbation}_{language}"].replace("{{original}}", sentence).replace("{{target_lang}}", target_lang)
                    print(prompt)
                    response = call_chatgpt_turbo(prompt)
                    print("> ", response)
                    print("=" * 80)
                    
                    data["perturbation"] = perturbation
                    data[f"pert_{language}"] = response
                    out_file.write(json.dumps(data, ensure_ascii=False) + "\n")