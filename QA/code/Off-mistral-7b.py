import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from QA.code.prompt import qa_prompt
from tqdm import tqdm

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input QG file")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sentence_type", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        total_examples = sum(1 for _ in f)

    with open(args.input_file, "r", encoding="utf-8") as f_in, \
         open(args.output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, total=total_examples, desc=f"Mistral QA ({args.sentence_type})"):
            data = json.loads(line)

            # Recupera il contesto nella lingua specifica (en, es, fr)
            sentence = data.get(args.sentence_type)
            questions = data.get("questions")

            if not sentence or not questions:
                continue

            # Il prompt rimane in inglese, ma 'sentence' sarà in spagnolo/francese!
            # Questo forza il modello a fare Cross-Lingual reasoning.
            prompt = (
                qa_prompt
                .replace("{{sentence}}", sentence)
                .replace("{{questions}}", questions)
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            generated = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()

            data["answers"] = generated
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
