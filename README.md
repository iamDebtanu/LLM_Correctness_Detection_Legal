# Peeking Inside LLMs: Leveraging Internal Artifacts of LLMs for Enhancing Reliability in Legal Classification
This repository contains code for:

1. Running inference with Large Language Models (LLMs)
2. Extracting hidden-state activations from transformer layers
3. Training a hallucination / correctness detector classifier
4. Performing hallucination-aware inference on legal datasets

The project supports multiple instruction-tuned LLMs such as:

- Meta-Llama-3.1-8B-Instruct
- Mistral-7B-Instruct
- Gemma-7B-It
- Qwen2.5-7B-Instruct

Supported datasets:

- ILDC (Indian Legal Documents Corpus)
- ECHR (European Court of Human Rights)

---

# 📂 Repository Structure

```text
.
├── llm_inference.py          # Extract hidden activations from LLMs
├── Store_classifier.py      # Train hallucination/correctness classifier
├── llm_HD_inference.py      # Hallucination-aware inference
├── requirements.txt
├── Datasets/
│   ├── ILDC_train.csv
│   ├── ILDC_test.csv
│   └── test_ECHR_Dataset_binary.csv
├── saved_model/
├── results_llm/
└── results_llm_HD/
```

---

# ⚙️ Environment Setup

## 1. Clone Repository

```bash
git clone https://github.com/iamDebtanu/LLM_Hallu_Legal.git
cd LLM_Hallu_Legal
```

---

## 2. Create Virtual Environment

### Linux / Mac

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---


# Hugging Face Access

Some models (especially Llama models) require Hugging Face authentication.

## Login

```bash
huggingface-cli login
```

Then enter your Hugging Face access token.

You must also request access to gated models such as:

- meta-llama/Meta-Llama-3.1-8B-Instruct

---

# 📁 Dataset Format

## ILDC Dataset

CSV columns:

```text
text,label
```

Where:

- `text` = legal case proceeding
- `label`:
  - `1` → Accept
  - `0` → Reject

---

## ECHR Dataset

CSV columns:

```text
text,binary_judgement
```

Where:

- `1` → Violation
- `0` → No-Violation

---

# Step 1 — Run LLM Inference

This step:

- Loads the dataset
- Runs inference using the selected LLM
- Extracts hidden activations from:
  - MLP layers
  - Attention layers
- Saves outputs for classifier training

---

## Basic Command

```bash
python llm_inference.py
```

Default configuration:

| Argument | Default |
|---|---|
| dataset_name | ILDC |
| model_name | Meta-Llama-3.1-8B-Instruct |
| iteration | 0 |
| interval | 2500 |

---

## Run on ILDC Dataset

```bash
python llm_inference.py \
    --dataset_name ILDC \
    --model_name Meta-Llama-3.1-8B-Instruct
```

---

## Run on ECHR Dataset

```bash
python llm_inference.py \
    --dataset_name ECHR \
    --model_name Qwen2.5-7B-Instruct
```

---

## Chunked Processing

Useful for large datasets or multi-GPU execution.

Example:

```bash
python llm_inference.py \
    --iteration 2 \
    --interval 1000
```

This processes:

```text
start = 2 * 1000 = 2000
end   = 3000
```

---

# 📤 Output of Step 1

Results are stored in:

```text
results_llm/<DATASET_NAME>/
```

Each saved file contains:

- model predictions
- logits
- correctness labels
- attention activations
- feed-forward activations

---

# Step 2 — Train Hallucination Classifier

This script trains a feed-forward classifier on extracted hidden states.

The classifier predicts whether the LLM prediction is likely correct or hallucinated.

---

## Run Training

```bash
python Store_classifier.py
```

---

## What the Classifier Uses

Possible feature types:

- middle attention layers
- last attention layers
- middle feed-forward layers
- last feed-forward layers

The classifier architecture:

```text
Input → Linear(256) → ReLU → Dropout → Linear(2)
```

---

# 💾 Saved Models

Trained classifiers are stored inside:

```text
saved_model/
```

Example:

```text
saved_model/saved_model_qwen_ILDC/
```

Saved checkpoint contains:

- model weights
- normalization statistics
- input dimensions

---

# Step 3 — Hallucination-Aware Inference

This script:

1. Runs normal LLM inference
2. Extracts hidden states
3. Uses the trained classifier to detect hallucinations
4. Modifies the final prediction based on hallucination confidence

---

## Run Command

```bash
python llm_HD_inference.py \
    --dataset_name ILDC \
    --model_name Qwen2.5-7B-Instruct
```

---

# Important Arguments

| Argument | Description |
|---|---|
| dataset_name | ILDC or ECHR |
| model_name | LLM to use |
| feature_type | Hidden-state feature selection |
| decision_mode | REF or REV |
| max_new_tokens | Maximum generated tokens |

---

## Feature Types

```text
mid3_fc
last3_fc
mid3_att
last3_att
```

Meaning:

- `mid3` = middle 3 transformer layers
- `last3` = final 3 transformer layers
- `fc` = feed-forward activations
- `att` = attention activations

---

# Decision Modes

## REF Mode

If hallucination is detected:

```text
Return: "Not Sure"
```

---

## REV Mode

If hallucination is detected:

```text
Return opposite prediction
```

Example:

```text
Accept → Reject
Violation → No-Violation
```

---

# 📤 Output of Hallucination-Aware Inference

Stored in:

```text
results_llm_HD/
```

The output includes:

- original LLM prediction
- hallucination score
- corrected prediction
- hidden-state features

---


# ⚠️ Important Notes

## 1. Update Dataset Paths

Inside scripts, verify dataset paths:

```python
"csv_path": "Datasets/ILDC_train.csv"
```

Modify them if your dataset location is different.

---

## 2. GPU Selection

The scripts currently use:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

Change this if using another GPU.

---

