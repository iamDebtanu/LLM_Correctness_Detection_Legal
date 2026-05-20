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
├── Store_classifier.py      # Train CD classifier
├── llm_HD_inference.py      #  LLM+CD inference
├── requirements.txt
├── Datasets/
│   ├── ILDC_train.csv
│   ├── ILDC_test.csv
│   |── ECHR_train.csv
|   └── ECHR_test.csv
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
Two files are generated for each run.

---

## 1. JSON File

Contains:

```text
- sample id
- question / input text
- gold answer
- model prediction
- correctness 
```

Example:

```json
{
  "id": 12,
  "question": "<legal text>",
  "gold_answer": "Accept",
  "prediction": "Reject",
  "correct_label": false
}
```

Where:

- `correct_label = true` → prediction correct
- `correct_label = false` → prediction incorrect / hallucinated

---

## 2. Pickle File (.pkl)

Contains extracted hidden activations such as:

```text
- feed-forward (fc) activations
- attention (att) activations
- logits
- hidden representations
```

These features are later used to train the hallucination classifier.

---

# Step 2 — Train CD Classifier

This script trains a feed-forward classifier on extracted hidden states.

The classifier predicts whether the LLM prediction is likely correct or hallucinated.

---

# ⚠️ Before Running Step 2

Inside `Store_classifier.py`, update:

## 1. Input File Paths

Change paths for:

```python
json_file_path = "results_llm/...json"
pickle_file_path = "results_llm/...pkl"
```

These files are generated from Step 1.

---

## 2. Save Directory

Update classifier save directory:

```python
save_dir = "saved_model/..."
```

Example:

```python
save_dir = "saved_model/saved_model_qwen_ILDC"
```

---

# ▶️ Run Training

Basic command:

```bash
python Store_classifier.py
```

---

# 🔧 Important Argument Parsers

The training script supports several argument parsers for flexible experiments.

---

## 1. Dataset Selection

Choose which dataset to train on.

```bash
python Store_classifier.py \
    --dataset_name ILDC
```

Possible values:

```text
ILDC
ECHR
```

---

## 2. Training Mode

Controls how hidden representations are used.

```bash
python Store_classifier.py \
    --dataset_name ILDC \
    --mode all
```

Possible values:

```text
all
single
multi
average
mid3
last3
```

Meaning:

| Mode | Description |
|---|---|
| all | Use all transformer layers |
| single | use one specific layer |
| multi | use selected multiple layers |
| average | use Average representations across layers |
| mid3 | Use middle 3 layers |
| last3 | Use final 3 layers |

---

## 3. Single Layer Training

Used with:

```text
--mode single
```

Example:

```bash
python Store_classifier.py \
    --dataset_name ILDC \
    --mode single \
    --layer 12
```

This trains classifier using only layer 12 representation.

---

## 4. Multiple Layer Training

Used with:

```text
--mode multi
```

Example:

```bash
python Store_classifier.py \
    --dataset_name ILDC \
    --mode multi \
    --layers 8 9 10
```

This combines representations from layers 8, 9, and 10.

---

## 5. Representation Type

Choose hidden representation type.

```bash
python Store_classifier.py \
    --dataset_name ILDC \
    --representation fc
```

Possible values:

```text
fc
att
```

Meaning:

| Representation | Description |
|---|---|
| fc | Feed-forward activations |
| att | Attention activations |

---

## 6. Full Example Commands

### Example A — Last 3 Feed-Forward Layers

```bash
python Store_classifier.py \
    --dataset_name ILDC \
    --mode last3 \
    --representation fc
```

---

### Example B — Single Attention Layer

```bash
python Store_classifier.py \
    --dataset_name ECHR \
    --mode single \
    --layer 15 \
    --representation att
```

---

### Example C — Multiple FC Layers

```bash
python Store_classifier.py \
    --dataset_name ILDC \
    --mode multi \
    --layers 5 6 7 8 \
    --representation fc
```

# Step 3 — LLM+CD Inference

This script:

1. Runs normal LLM inference
2. Extracts hidden states
3. Uses the trained classifier to detect correctness
4. Modifies the final prediction based on correctness confidence

---
# ▶️ Basic Run

```bash
python llm_HD_inference.py
```

# ▶️ Example Commands with Different Arguments

Before running Step 3, update the following paths inside:

```text
llm_HD_inference.py
```

---

# ⚠️ Important Path Configuration

## 1. Dataset Path

Update dataset CSV path:

```python
csv_path = "Datasets/ILDC_test.csv"
```

Examples:

```python
csv_path = "Datasets/ILDC_test.csv"
```

---

## 2. Classifier Model Path

Choose the trained classifier checkpoint from Step 2.

Example:

```python
classifier_path = "saved_model/saved_model_qwen_ILDC/model.pth"
```

---

## 3. Results Save Directory

Update output directory:

```python
save_dir = "results_llm_HD/"
```

Example:

```python
save_dir = "results_llm_HD/qwen_ILDC"
```

---

# ⚠️ Important Note About Feature Types

The feature type used in Step 3 must match the classifier trained in Step 2.

Example:

If classifier was trained using:

```bash
--mode last3 --representation fc
```

then Step 3 should also use:

```bash
--feature_type last3_fc
```

Otherwise, feature dimensions will not match.

---

## Example 1 — Run on ILDC

```bash
python llm_HD_inference.py \
    --dataset_name ILDC \
    --model_name Qwen2.5-7B-Instruct \
    --feature_type last3_fc
```

---

## Example 2 — Run on ECHR Dataset

```bash
python llm_HD_inference.py \
    --dataset_name ECHR \
    --model_name Meta-Llama-3.1-8B-Instruct \
    --feature_type mid3_att
```

---

## Example 3 — Use Different Feature Type

```bash
python llm_HD_inference.py \
    --feature_type last3_att
```

Possible values:

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
- `att` = attention activation
---

## Example 4 — Use REF Decision Mode

```bash
python llm_HD_inference.py \
    --decision_mode REF
```

Behavior:

```text
if Hallucination detected → Return "Not Sure"
```

---

## Example 5 — Use REV Decision Mode

```bash
python llm_HD_inference.py \
    --decision_mode REV
```

Behavior:

```text
if Hallucination detected → Reverse original prediction which is predicted by llm
```

---

## Example 6 — Change Maximum Generation Length

```bash
python llm_HD_inference.py \
    --max_new_tokens 5
```

---

## Example 7 — Full Example

```bash
python llm_HD_inference.py \
    --dataset_name ILDC \
    --model_name Qwen2.5-7B-Instruct \
    --feature_type last3_fc \
    --decision_mode REV \
    --max_new_tokens 5
```
# 📤 Output of LLM+CD Inference

Stored in:

```text
results_llm_HD/
```

Step 3 generates a JSON file containing:

```text
- sample id
- question / input text
- gold answer
- LLM+CD prediction
```

Example:

```json
{
  "id": 21,
  "question": "<legal text>",
  "gold_answer": "Accept",
  "LLM+CD prediction": "Not Sure"
}
```


# ⚠️ Important Notes


## GPU Selection

The scripts currently use:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

Change this if using another GPU.

---

