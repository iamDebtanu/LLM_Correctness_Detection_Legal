# 🧠 Hallucination Detection – LLM Hidden State Attribution

This repository contains **Hallucination.py**, a script designed to collect hidden activations, attention outputs, and Integrated Gradients explanations from large language models (LLaMA, Falcon, OPT).  
It supports multiple datasets and chunked execution for large-scale hallucination analysis.

---

## 🚀 Features

- Supports LLaMA, Falcon, and OPT models  
- Yes/No QA with strict question templates  
- Works with multiple datasets (`custom_qa`, `capitals`, `place_of_birth`, `trivia_qa`)  
- Extracts:  
  - Fully connected (MLP) activations  
  - Attention layer activations  
  - Integrated Gradients attributions  
- Chunked data processing: `--iteration`, `--interval`  
- Saves results as `.pickle` files  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/hallucination-detection.git
cd hallucination-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Required packages:
```
torch
transformers
datasets
pandas
numpy
tqdm
accelerate
sentencepiece
captum
```

---

## 📘 Usage

Run the script using **argparse** command-line arguments.

### ✅ Basic Run
```bash
python Hallucination.py
```

Defaults:
- `model_name = open_llama_7b`
- `dataset_name = custom_qa`
- `iteration = 0`
- `interval = 2500`

---

## 🔧 Select a Model

```bash
python Hallucination.py --model_name open_llama_13b
```

Supported models:
- falcon-40b  
- falcon-7b  
- open_llama_13b  
- open_llama_7b  
- opt-6.7b  
- opt-30b  

---

## 📂 Select a Dataset

```bash
python Hallucination.py --dataset_name capitals
```

Dataset options:

| dataset_name      | description                               |
|-------------------|-------------------------------------------|
| custom_qa         | CSV with Question + Answer                |
| capitals          | CSV subject → capital                     |
| place_of_birth    | CSV subject → birthplace                  |
| trivia_qa         | HuggingFace Trivia-QA dataset             |

---

## 🔁 Chunked Execution (for parallel / large datasets)

Example: process 3rd chunk of size 2500
```bash
python Hallucination.py --iteration 3 --interval 2500
```

This processes rows:
```
start = 3 * 2500 = 7500
end   = 10000
```

---

## 📝 Input File Format

### ✔ custom_qa.csv
```
Question,Answer
"Is sky blue?",Yes
"What is 2+2?",4
```

### ✔ capitals.csv / place_of_birth.csv
```
subject,object
Germany,Berlin
India,New Delhi
```

---

## 📁 Output Files

All results are saved in:

```
./results/
```

File naming format:
```
<model>_<dataset>_start-<x>_end-<y>_<month>_<day>.pickle
```

Each pickle contains:
- question  
- answers  
- generated response (tokens + string)  
- logits  
- correctness flag  
- MLP activations  
- Attention activations  
- Integrated Gradients attribution  

---




