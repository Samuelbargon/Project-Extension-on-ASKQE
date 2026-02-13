import json
import argparse
import os
import csv
import re


def tokenize(text):
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def load_data(file_path, context_key):
    """Load (answer, context) pairs from a QA jsonl file."""
    pairs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            context = data.get(context_key, "")
            answers = data.get("answers", [])

            if not context or not answers:
                continue

            for ans in answers:
                if ans:
                    pairs.append((ans, context))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5-base")
    parser.add_argument(
        "--output_file",
        default="evaluation/answer_context_overlap/overlap_t5-base.csv"
    )
    args = parser.parse_args()

    pipelines = ["vanilla", "atomic", "semantic"]
    context_keys = {
        "vanilla": "en",
        "atomic": "atomic_facts",
        "semantic": "semantic_roles"
    }

    results = []

    for pipeline in pipelines:
        qa_file = f"QA/{args.model}/en-{pipeline}.jsonl"

        if not os.path.exists(qa_file):
            print(f"[SKIP] File not found: {qa_file}")
            continue

        print(f"Evaluating answer–context overlap for {pipeline}...")

        pairs = load_data(qa_file, context_keys[pipeline])
        if not pairs:
            print(f"[WARNING] No valid data for {pipeline}")
            continue

        scores = []

        for answer, context in pairs:
            ans_tokens = tokenize(answer)
            ctx_tokens = tokenize(context)

            if not ans_tokens:
                continue

            overlap = len(set(ans_tokens) & set(ctx_tokens)) / len(set(ans_tokens))
            scores.append(overlap)

        avg_score = sum(scores) / len(scores)

        results.append({
            "model": args.model,
            "pipeline": pipeline,
            "num_samples": len(scores),
            "avg_context_overlap": avg_score
        })

        print(f"  → avg overlap: {avg_score:.4f}")

    with open(args.output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "pipeline", "num_samples", "avg_context_overlap"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {args.output_file}")


if __name__ == "__main__":
    main()
