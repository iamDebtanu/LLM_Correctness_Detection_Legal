# 🧠 Hallucination Detection – LLM Hidden State Attribution

This repository contains **Hallucination.py**, a script designed to collect hidden activations, attention outputs, and Integrated Gradients explanations from large language models (LLaMA, Falcon, OPT).  
It supports multiple datasets and chunked execution for large-scale hallucination analysis.

---

## 🚀 Features

- Supports LLaMA, Falcon, and OPT models  
- Yes/No QA with strict question templates  
- Works with multiple datasets (`ECHR`, `ILDC`)  
- Extracts:  
  - Fully connected (MLP) activations  
  - Attention layer activations   
- Chunked data processing: `--iteration`, `--interval`  
- Saves results as `.pickle` and `json` files  

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


---

## 📘 Usage

Run the script using **argparse** command-line arguments.

### ✅ Basic Run
```bash
python llm_Inference_code/llm_Inference_ECHR.py
```

Defaults:
- `model_name = Meta-Llama-3.1-8B-Instruct`
- `iteration = 0`
- `interval = 2500`

---

## 🔧 Select a Model

```bash
python llm_Inference_code/llm_Inference_ECHR.py --model_name mistral-7b-instruct-v0.3
```

Supported models:
- mistral-7b-instruct-v0.3  
- Meta-Llama-3.1-8B-Instruct 
- gemma-7b-it  

---

## 📂 Select a Dataset


Dataset options:

| dataset_name      | description                               |
|-------------------|-------------------------------------------|
| ILDC         | A dataset of Indian Supreme Court cases for legal judgment prediction (accept/reject decisions) |
| ECHR          | A dataset of European Court of Human Rights cases used for predicting legal outcomes (violation vs. non-violation). |
---

## 🔁 Chunked Execution (for parallel / large datasets)

Example: process 3rd chunk of size 2500
```bash
python llm_Inference_code/llm_Inference_ECHR.py --iteration 3 --interval 2500
```

This processes rows:
```
start = 3 * 2500 = 7500
end   = 10000
```

---

## 📝 Input File Format

### ✔ ILDC dataset
```
text,label
"The appellant challenged the judgment...",1
"The court found no merit in the appeal...",0
```

### ✔ ECHR dataset
```
text,label
"The applicant alleged a violation of rights...",1
"No violation was found by the court...",0
```

---

## 🧠 Mid-Layer Selection (LLM + HD)

For extracting **mid-layer representations (mid3)** in the LLM+HD setup, we select three consecutive layers from the middle of the model.

### ✔ Layer Selection Strategy

- For models with **28 layers** (e.g., Gemma):
  [12, 13, 14]

- For models with **32 layers** (e.g., LLaMA, Mistral):
  [14, 15, 16]



