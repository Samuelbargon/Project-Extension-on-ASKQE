import json
import argparse
import re
import numpy as np


def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def load_data(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            context = data.get("en", "")
            answers = data.get("answers", [])
            if isinstance(answers, list):
                answers = " ".join(answers)
            examples.append((context, answers))
    return examples


def compute_overlap(examples):
    scores = []
    for context, answer in examples:
        a_tokens = tokenize(answer)
        c_tokens = tokenize(context)

        if len(a_tokens) == 0:
            scores.append(0.0)
        else:
            overlap = len(a_tokens & c_tokens) / len(a_tokens)
            scores.append(overlap)

    return np.mean(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral", required=True)
    parser.add_argument("--llama", required=True)
    args = parser.parse_args()

    mistral_data = load_data(args.mistral)
    llama_data = load_data(args.llama)

    assert len(mistral_data) == len(llama_data)

    mistral_score = compute_overlap(mistral_data)
    llama_score = compute_overlap(llama_data)

    print(f"Mistral Answer–Context Overlap: {mistral_score:.4f}")
    print(f"LLaMA  Answer–Context Overlap: {llama_score:.4f}")


if __name__ == "__main__":
    main()
