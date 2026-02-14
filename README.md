# ASKQE Extension with Mistral-7B
This repository extends the ASKQE (Question Answering as Automatic Evaluation) framework using Mistral-7B-Instruct-v0.3 to evaluate translation quality at the answer level.
Instead of comparing sentences directly, the pipeline measures whether translated texts preserve meaning by comparing answers generated from the source and translated contexts.

The files added are:
1. Official_Notebook_mistral_7b.ipynb:
   - Monolingual English QA baseline
   - Model: Mistral-7B-Instruct-v0.3 (zero-shot)
   - 972 examples
   - Input: English context + generated questions
   - Output: List of answers (greedy decoding)
   - Evaluation: SBERT embeddings + cosine similarity
   
   This notebook establishes the English reference QA outputs.

2. build_clean_crosslingual_dataset.ipynb:
   - Builds a clean EN→ES aligned dataset
   - Translates English contexts into Spanish
   - Keeps original English questions fixed
   - No perturbations

   Used as baseline cross-lingual condition

3. ASKQE_and_backtranslation.ipynb:

   Implements two evaluation settings:
   - Backtranslation pipeline: EN → ES → EN → QA → comparison
   - Direct cross-lingual QA: QA on Spanish context with English questions (no backtranslation)

   Both settings are compared against the English QA baseline using SBERT cosine similarity

4. ASKQE_and_backtranslation_perturbations.ipynb:
      Evaluates translation robustness under controlled perturbations:
      - Alteration
      - Synonym substitution
      - Omission
      - Word order
      - Intensifier

      Perturbations are applied to Spanish translations.
      QA is performed with Mistral-7B and outputs are stored in QA/mistral-7b/es_noise/

5. Evaluation_perturbations.ipynb:
      - Performs quantitative evaluation of perturbed outputs
      - Compares answers from noisy translations to English baseline.
      - Uses SBERT cosine similarity and measures answer-level semantic degradation


# AskQE: Data Preprocessing & Language Expansion
To add new languages and prepare the data, run the following two scripts in order.

### 1. Data Ingestion (Format Conversion)
Converts raw TICO-19 `.tsv` files into the AskQE compatible `.jsonl` format.
```bash
python ./data/code/convert.py
```

### 2. Standardization & Project Configuration
Run this command to validate files and update the global LANGUAGES list across the project's 11 dependent files.
```bash
python ./data/code/config_raw_data.py
```

If the script identifies a language code it cannot resolve (an "Unresolved Mismatch"), you may use --clean to automatically delete any files that cannot be standardized
```bash
python ./data/code/config_raw_data.py --clean
```
