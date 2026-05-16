# 🧠 Correctness Detection – LLM Hidden State Attribution

This repository contains **llm_Inference_ECHR.py** and **llm_Inference_ILDC.py**, scripts designed to collect hidden activations and attention outputs from large language models (LLaMA, Mistral, Qwen) for ECHR and ILDC datasets. The extracted internal representations are subsequently used to train correctness detectors (CDs). The training scripts and related classifier codes are provided in the `store_classifiers_all` directory. The trained CDs are applied to identify potentially incorrect predictions. The corresponding LLM+CD inference pipelines are available in the `llm_HD_Inference_code` folder.
It supports chunked execution for large-scale correctness analysis.

---

## 🚀 Features

- Supports LLaMA, Mistral, and Qwen models  
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
git clone https://github.com/iamDebtanu/LLM_Hallu_Legal.git
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
- Qwen2.5-7B-Instruct  

---

## 📂 Select a Dataset


Dataset options:

| dataset_name      | description                               |
|-------------------|-------------------------------------------|
| ILDC         | A dataset of Indian Supreme Court cases for legal judgment prediction (accept/reject decisions) |
| ECHR          | A dataset of European Court of Human Rights cases used for predicting legal outcomes (violation vs. no-violation). |
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
"The appellant challenged the judgment...",1(accept)
"The court found no merit in the appeal...",0(reject)
```

### ✔ ECHR dataset
```
text,label
"The applicant alleged a violation of rights...",1(violation)
"No violation was found by the court...",0(no-violation)
```

---

## 🧠 Mid-Layer Selection (LLM + HD)

For extracting **mid-layer representations (mid3)** in the LLM+HD setup, we select three consecutive layers from the middle of the model.

### ✔ Layer Selection Strategy

- For models with **28 layers** (e.g., Qwen):
  [12, 13, 14]

- For models with **32 layers** (e.g., LLaMA, Mistral):
  [14, 15, 16]



