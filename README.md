# Peeking Inside LLMs: Leveraging Internal Artifacts of LLMs for Enhancing Reliability in Legal Classification
Long Paper accepted at `ASAIL 2026!`
This repository contains scripts to extract hidden-state activations from instruction-tuned LLMs, train a Correctness Detector (CD) classifier, and run hallucination-aware inference on legal datasets.

Supported instruction-tuned LLMs: `Meta-Llama-3.1-8B-Instruct`,`Mistral-7B-Instruct`,`Qwen2.5-7B-Instruct`

Supported datasets: `ILDC` (Labels: 1=Accept, 0=Reject), `ECHR` (Labels: 1=Violation, 0=No-Violation)

---

# 📂 Repository Structure

```text
.
├── llm_inference.py         # Extract hidden activations from LLMs
├── Store_classifier.py      # Train CD classifier and store
├── llm_HD_inference.py      #  LLM+CD hallucination-aware inference
├── requirements.txt
├── Datasets/
│   ├── ILDC_train.csv
│   ├── ILDC_test.csv
│   |── ECHR_train.csv
|   └── ECHR_test.csv
├── saved_model/             # Trained CD models saved here
├── results_llm/             # llm outputs
└── results_llm_HD/          # llm+CD outputs
```

---

# ⚙️ Environment Setup

## 1. Clone Repository

```bash
git clone https://github.com/iamDebtanu/LLM_Hallu_Legal.git
cd LLM_Hallu_Legal
```

## 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login
```
 Note: Access approval is required via Hugging Face for gated models like `meta-llama/Meta-Llama-3.1-8B-Instruct`
# Execution Pipeline

## Step 1 — Run LLM Inference & Feature Extraction

Extracts hidden-state activations(`.pkl`) and LLM outputs (`.json`).

```bash
# Basic 
python llm_inference.py--dataset_name ILDC
# With custom model and dataset
python llm_inference.py --dataset_name ECHR --model_name Qwen2.5-7B-Instruct
```
Use `--iteration` and `--interval` to chunk process large datasets across multiple runs or GPUs

Output: Stored in `results_llm/<DATASET_NAME>/`

## Step 2 — Train CD Classifier

Trains a feed-forward classifier on extracted hidden states to predicts whether the LLM prediction is likely correct or hallucinated.

```bash
# ⚠️ FIRST: Open Store_classifier.py and update input JSON/PKL paths & save_dir.
python Store_classifier.py --dataset_name ILDC --mode last3 --representation fc
```
Key Arguments:

| Argument           | Options                                              | Description                              |
| ------------------ | ---------------------------------------------------- | ---------------------------------------- |
| `--mode`           | `all`, `single`, `multi`, `average`, `mid3`, `last3` | Layer selection strategy                 |
| `--representation` | `fc`, `att`                                          | Feed-forward or Attention representation |
| `--layer`          | `int`                                                | Layer index for `single` mode            |
| `--layers`         | `multiple ints`                                      | Multiple layer indices for `multi` mode  |

where `all` -> Use all transformer layers, `single` -> use one specific layer, `multi` -> use selected multiple layers, `average` -> use average representations across layers, `mid3` -> Use average middle 3 layers and `last3` -> Use average final 3 layers.

Examples:
```bash
# Average of last 3 feed-forward layers
python Store_classifier.py --dataset_name ILDC --mode last3 --representation fc

# Single attention layer
python Store_classifier.py --dataset_name ECHR --mode single --layer 15 --representation att
```

# Step 3 — Hallucination-Aware Inference (LLM + CD)

Integrates the base LLM with trained CD classifier to flag or fix uncertain answers.

---

```bash
# ⚠️ FIRST: Open llm_HD_inference.py and update csv_path, classifier_path, and save_dir.
python llm_HD_inference.py --dataset_name ILDC --model_name Qwen2.5-7B-Instruct --feature_type last3_fc --decision_mode REF
```
- `--feature_type` : Must match Step 2 training configuration (mid3_fc, last3_fc, mid3_att, last3_att).
- `--decision_mode` : `REF` (return "Not Sure") or `REV` (reverse prediction)
- `--max_new_tokens` : Adjust maximum generated token during the execution pass (e.g., `--max_new_tokens 5`).

Output: Stored in `results_llm_HD/`
