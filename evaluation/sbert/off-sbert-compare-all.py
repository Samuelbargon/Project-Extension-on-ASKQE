import os
import json
import csv
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util


def load_answers(jsonl_path):
    """
    Load model answers from a JSONL file.
    If answers are stored as a list, they are joined into a single string.
    """
    answers = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            ans = data.get("answers", "")

            if isinstance(ans, list):
                ans = " ".join(ans)

            answers.append(ans)

    return answers


def main():
    parser = argparse.ArgumentParser(description="SBERT-based comparison of QA model outputs")

    parser.add_argument(
        "--reference_model",
        type=str,
        default="mistral-7b",
        help="Name of the reference model"
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        required=True,
        help="Path to the reference model JSONL output"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="QA",
        help="Root directory containing model outputs"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["vanilla", "atomic", "semantic"],
        required=True,
        help="Question type to evaluate"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation/sbert/results.csv",
        help="CSV file where results will be appended"
    )

    args = parser.parse_args()

    # Models to compare against the reference
    compared_models = [
        "gemma-9b",
        "gemma-27b",
        "llama-8b",
        "llama-70b",
        "yi-9b",
    ]

    print("Loading SBERT model...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading reference answers...")
    reference_answers = load_answers(args.reference_file)

    csv_exists = os.path.isfile(args.output_csv)

    with open(args.output_csv, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Write header only once
        if not csv_exists:
            writer.writerow([
                "reference_model",
                "compared_model",
                "pipeline",
                "num_examples",
                "avg_sbert_similarity"
            ])

        for model_name in compared_models:
            compare_file = os.path.join(
                args.models_dir,
                model_name,
                f"en-{args.pipeline}.jsonl"
            )

            if not os.path.isfile(compare_file):
                print(f"[SKIP] Missing file for {model_name}")
                continue

            print(f"\nComparing {args.reference_model} vs {model_name}")
            compared_answers = load_answers(compare_file)

            if len(reference_answers) != len(compared_answers):
                print(f"[SKIP] Length mismatch for {model_name}")
                continue

            similarities = []

            for ref_ans, cmp_ans in zip(reference_answers, compared_answers):
                emb_ref = sbert.encode(ref_ans, convert_to_tensor=True)
                emb_cmp = sbert.encode(cmp_ans, convert_to_tensor=True)

                sim_score = util.cos_sim(emb_ref, emb_cmp).item()
                similarities.append(sim_score)

            avg_similarity = float(np.mean(similarities))
            print(f"Average SBERT similarity: {avg_similarity:.4f}")

            writer.writerow([
                args.reference_model,
                model_name,
                args.pipeline,
                len(reference_answers),
                f"{avg_similarity:.6f}"
            ])


if __name__ == "__main__":
    main()
