# Peeking Inside LLMs: Leveraging Internal Artifacts of LLMs for Enhancing Reliability in Legal Classification
Long Paper accepted at **Workshop on Automated Semantic Analysis of Information in Law (ASAIL 2026)**, co-located with the **International Conference on Artificial Intelligence and Law (ICAIL 2026)**. This repository contains the scripts developed in this work.

## 📂 Repository Structure
```text
├── scripts/
│   ├── llm_inference.py         # Extract hidden activations and LLM predictions
│   ├── Store_classifier.py      # Train and save the CD classifier
│   └── llm_CD_inference.py      # Performing inference of LLM+CD system
├── requirements.txt             # Required libraries for running the repository
```

## ⚙️ Environment Setup
```bash
# Clone the repository
git clone https://github.com/iamDebtanu/LLM_Correctness_Detection_Legal.git
cd LLM_Correctness_Detection_Legal
# Create and activate a virtual environment
conda create -n llmcd python=3.12
conda activate llmcd
# Install required dependencies
pip install -r requirements.txt
# HuggingFace Authentication (HuggingFace access approval is required for gated models like `meta-llama/Llama-3.1-8B-Instruct`)
huggingface-cli login
```

# Execution Pipeline

### Step 1: LLM Inference & Artifacts Extraction

Runs the selected LLM on the dataset and extracts internal artifacts for CD training
```bash
# With custom model and dataset
python llm_inference.py --dataset_name ECHR --model_name Qwen2.5-7B-Instruct
```

| Argument         | Options                                                                  | Description                            |
|------------------|--------------------------------------------------------------------------|----------------------------------------|
| `--dataset_name`  | `ILDC`, `ECHR` | Dataset to run inference on. ([ILDC Train](https://huggingface.co/datasets/Exploration-Lab/IL-TUR/viewer/cjpe/single_train), [ILDC Test](https://huggingface.co/datasets/Exploration-Lab/IL-TUR/viewer/cjpe/test), [ECHR Dataset](https://archive.org/download/ECHR-ACL2019))                                                                                           |
| `--model_name`   | `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct`, `Qwen2.5-7B-Instruct` | Model used for inference in this work.|
| `--iteration`    | Integer (e.g., `0`, `1`, `2`, ...)                                       | Chunk index for parallel execution (applicable for large dataset).|
| `--interval`     | Positive integer (e.g., `2500`)                                          | Number of samples processed per chunk. |

**Output:** Saves generated LLM outputs in `.json` files and extracted internal artifacts in `.pkl` files under `results_llm/<dataset_name>/`.

### Step 2: Train CD Classifier

Trains a feed-forward classifier on extracted hidden states to predict whether the LLM prediction is likely correct or incorrect.
```bash
# Open Store_classifier.py file and update input (json/pkl) & save_dir paths.
python Store_classifier.py --dataset_name ILDC --mode last3 --representation fc
# Average of middle-3 feed-forward layers
python Store_classifier.py --dataset_name ILDC --mode mid3 --representation fc
# Single attention layer
python Store_classifier.py --dataset_name ECHR --mode single --layer 15 --representation att
```

| Argument           | Options                                              | Description                              |
| ------------------ | ---------------------------------------------------- | ---------------------------------------- |
| `--mode`           | `all`, `single`, `multi`, `average`, `mid3`, `last3` | Layer selection strategy                 |
| `--representation` | `fc`, `att`                                          | Feed-forward or Attention representation |
| `--layer`          | `int`                                                | Layer index for `single` mode            |
| `--layers`         | `multiple ints`                                      | Multiple layer indices for `multi` mode  |

The `--mode` argument determines how layer representations are extracted: `all` (all layers), `single` (one selected layer), `multi` (multiple selected layers), `average` (average of specified layers), `mid3` (average of the middle three layers), and `last3` (average of the final three layers).

**Output:** Saves the trained CD classifier as a `.pth` file and stores the corresponding AUROC score in a `.txt` file in `save_dir`.

### Step 3: Reliability Assessment and Final Decision (LLM + CD)

Integrates the base LLM with a trained CD classifier to assess the reliability of the LLM predictions.
```bash
# Open llm_CD_inference.py file and update csv_path, classifier_path, and output_dir.
python llm_CD_inference.py --dataset_name ILDC --model_name Qwen2.5-7B-Instruct --feature_type last3_fc --decision_mode REF
```

| Argument | Options | Description |
|--------------|----------|-------------|
| `--dataset_name` | `ILDC`, `ECHR` | Dataset used for inference. |
| `--model_name` | `Llama-3.1-8B-Instruct`, `Mistral-7B-Instruct`, `Qwen2.5-7B-Instruct` | Base LLM used for prediction. |
| `--feature_type` | `mid3_fc`, `last3_fc`, `mid3_att`, `last3_att` | Feature representation used by the trained CD classifier. Must match Step 2 training configuration. |
| `--decision_mode` | `REF`, `REV` | CD strategy: `REF` returns **"Not Sure"** when the CD classifies as incorrect, while `REV` reverses the LLM prediction. |
| `--max_new_tokens` | Positive integer | Maximum number of tokens generated by the LLM. |

**Output:** Saves predictions generated by the LLM+CD framework in `.json` files in `output_dir`.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{santra2026llmcd, 
 title= "Peeking Inside {LLMs}: Leveraging Internal Artifacts of {LLMs} for Enhancing Reliability in Legal Classification", 
 author= "Santra, Sudipta and Datta, Debtanu and Ghosh, Saptarshi",
 booktitle= "Proceedings of the 8th Workshop on Automated Semantic Analysis of Information in Law co-located with the 21st International Conference on Artificial Intelligence and Law ({ICAIL} 2026)", 
 year= "2026"
 }
```
