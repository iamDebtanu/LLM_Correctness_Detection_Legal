# ============================================================
# IMPORTS
# ============================================================

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# GPU SETUP
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# FEED FORWARD CLASSIFIER (CD)
# ============================================================

class FFClassifier(torch.nn.Module):
    """
    Simple feed-forward classifier used for correctness detection.
    """

    def __init__(self, input_dim, dropout=0.1):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_name",
    type=str,
    choices=["ILDC", "ECHR"],
    required=True,
    help="Dataset name"
)

parser.add_argument(
    "--model_name",
    type=str,
    default="Meta-Llama-3.1-8B-Instruct",
    help="Model name from model_repos"
)

parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=5,
    help="Maximum number of generated tokens"
)

parser.add_argument(
    "--save_every",
    type=int,
    default=100,
    help="Save checkpoint after every N samples"
)

parser.add_argument(
    "--feature_type",
    type=str,
    default="last3_att",
    choices=["mid3_fc", "last3_fc", "mid3_att", "last3_att"],
    help="Feature extraction method for correctness detection"
)

parser.add_argument(
    "--decision_mode",
    type=str,
    default="REF",
    choices=["REF", "REV"],
    help=(
        "REF: return 'Not Sure' when hallucination detected. "
        "REV: return opposite class when hallucination detected."
    )
)

args = parser.parse_args()


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_CONFIG = {
    "ILDC": {
        "csv_path": "Datasets/ILDC_test.csv",
        "text_column": "text",
        "label_column": "label",
        "label_map": {
            1: "Accept",
            0: "Reject"
        },
        "classifier_path": "saved_model/saved_model_qwen_ILDC/att_mid3/mid3_layers.pth",
        "output_dir": "results_llm_HD/ILDC",
        "system_prompt": (
                    "You are a legal expert specialized in Indian law. "
                    "Your task is to analyze the given case proceeding strictly based on "
                    "principles of Indian law and established judicial reasoning, and determine "
                    "whether the appeal should be accepted or rejected. "
                    "Your decision must be solely based on the facts and legal issues presented "
                    "in the case. "
                    "Output only one word: either \"Accept\" or \"Reject\"."
                ),
        "question_template": (
            "Predict the outcome of the appeal based strictly on Indian law.\n\n"
            "Case Proceeding:\n{text}\n\n"
            "Decision:"
        )
    },

    "ECHR": {
        "csv_path": "Datasets/test_ECHR_Dataset_binary.csv",
        "text_column": "text",
        "label_column": "binary_judgement",
        "label_map": {
            1: "Violation",
            0: "No-Violation"
        },
        "classifier_path": "saved_model/saved_model_meta_ECHR/att_mid3/mid3_layers.pth",
        "output_dir": "results_llm_HD/ECHR",
        "system_prompt": (
                    "You are a legal expert specialized in human rights law under the "
                    "European Convention on Human Rights (ECHR). "
                    "Your task is to analyze case proceedings from the European Court "
                    "of Human Rights and determine whether the facts disclose a "
                    "violation of a human rights article under the ECHR. "
                    "Your decision must be strictly based on the facts provided and "
                    "the provisions of the European Convention on Human Rights. "
                    "Output only one word: either \"Violation\" or \"No-Violation\"."
                ),
        "question_template": (
            "Based on the European Convention on Human Rights, "
            "determine whether the following case proceeding "
            "shows a violation of a human rights article.\n\n"
            "Case Proceeding:\n{text}\n\n"
            "Decision:"
        )
    }
}

config = DATASET_CONFIG[args.dataset_name]


# ============================================================
# LOAD TRAINED HALLUCINATION CLASSIFIER
# ============================================================

HALLUCINATION_THRESHOLD = 0.0

ckpt = torch.load(
    config["classifier_path"],
    map_location=device,
    weights_only=False
)

hallucination_clf = FFClassifier(ckpt["input_dim"]).to(device)
hallucination_clf.load_state_dict(ckpt["state_dict"])
hallucination_clf.eval()

mean = ckpt["mean"].detach().cpu().numpy()
std = ckpt["std"].detach().cpu().numpy()


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPOS = {
    "Meta-Llama-3.1-8B-Instruct": (
        "meta-llama",
        r".*model.layers.[0-9]+.mlp.up_proj",
        r".*model.layers.[0-9]+.self_attn.o_proj"
    ),

    "mistral-7b-instruct-v0.3": (
        "mistralai",
        r".*model.layers.[0-9]+.mlp.up_proj",
        r".*model.layers.[0-9]+.self_attn.o_proj"
    ),

    "gemma-7b-it": (
        "google",
        r".*model.layers.[0-9]+.mlp.up_proj",
        r".*model.layers.[0-9]+.self_attn.o_proj"
    ),

    "Qwen2.5-7B-Instruct": (
        "qwen",
        r".*model.layers.[0-9]+.mlp.up_proj",
        r".*model.layers.[0-9]+.self_attn.o_proj"
    )
}


# ============================================================
# STORAGE FOR HIDDEN STATES
# ============================================================

attention_hidden_layers = defaultdict(list)
fully_connected_hidden_layers = defaultdict(list)


# ============================================================
# HOOK FUNCTIONS
# ============================================================


def save_fc_hidden(layer_name, module, inp, output):
    """
    Save only the hidden state of the final generated token from feed-forward layer.
    """

    last_token = output[:, -1, :].detach().to(torch.float16)
    fully_connected_hidden_layers[layer_name].append(
        last_token.cpu().numpy()
    )



def save_attention_hidden(layer_name, module, inp, output):
    """
    Save only the hidden state of the final generated token from attention layer.
    """

    last_token = output[:, -1, :].detach().to(torch.float16)
    attention_hidden_layers[layer_name].append(
        last_token.cpu().numpy()
    )


# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset_from_csv():
    """
    Load dataset and create question-answer pairs.
    """

    df = pd.read_csv(config["csv_path"])

    dataset = []

    for _, row in df.iterrows():

        question = config["question_template"].format(
            text=row[config["text_column"]]
        )

        answer = config["label_map"][row[config["label_column"]]]

        dataset.append((question, answer))

    return dataset[:20]


# ============================================================
# BUILD CHAT PROMPT
# ============================================================


def build_chat_prompt(question, tokenizer):
    """
    Create chat-formatted prompt.
    """

    messages = [
        {
            "role": "system",
            "content": config["system_prompt"]
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


# ============================================================
# TRUNCATE LONG INPUTS
# ============================================================


def smart_truncate(input_ids, max_tokens=8192):
    """
    Keep beginning and ending tokens if sequence is too long.
    """

    seq_len = input_ids.shape[1]

    if seq_len <= max_tokens:
        return input_ids

    keep_head = max_tokens // 2
    keep_tail = max_tokens - keep_head

    head = input_ids[:, :keep_head]
    tail = input_ids[:, -keep_tail:]

    return torch.cat([head, tail], dim=1)

# ============================================================
# GET MIDDLE INDICES
# ============================================================

def get_middle_indices(total_layers):
    """
    Return indices of middle 3 layers dynamically.

    Examples:
        32 layers -> [14, 15, 16]
        28 layers -> [12, 13, 14]
        60 layers -> [29, 30, 31]
    """

    mid = total_layers // 2

    return [mid - 2, mid - 1, mid]

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================


def get_mid3_fc_average():
    """
    Average hidden states from middle 3 feed-forward layers.
    """

    layer_vectors = []

    for key in fully_connected_hidden_layers:
        layer_vectors.append(
            fully_connected_hidden_layers[key][0].squeeze(0)
        )
    indices = get_middle_indices(len(layer_vectors))

    middle_vectors = [
        layer_vectors[i]
        for i in indices
    ]

    middle_vectors = layer_vectors[12:15]

    avg_vector = np.mean(middle_vectors, axis=0)

    return torch.tensor(
        avg_vector,
        dtype=torch.float32
    ).unsqueeze(0).to(device)



def get_last3_fc_average():
    """
    Average hidden states from last 3 feed-forward layers.
    """

    layer_vectors = []

    for key in fully_connected_hidden_layers:
        layer_vectors.append(
            fully_connected_hidden_layers[key][0].squeeze(0)
        )

    last_vectors = layer_vectors[-3:]

    avg_vector = np.mean(last_vectors, axis=0)

    return torch.tensor(
        avg_vector,
        dtype=torch.float32
    ).unsqueeze(0).to(device)



def get_mid3_attention_average():
    """
    Average hidden states from middle 3 attention layers.
    """

    layer_vectors = []

    for key in attention_hidden_layers:
        layer_vectors.append(
            attention_hidden_layers[key][0].squeeze(0)
        )

    indices = get_middle_indices(len(layer_vectors))

    middle_vectors = [
        layer_vectors[i]
        for i in indices
    ]

    avg_vector = np.mean(middle_vectors, axis=0)

    return torch.tensor(
        avg_vector,
        dtype=torch.float32
    ).unsqueeze(0).to(device)



def get_last3_attention_average():
    """
    Average hidden states from last 3 attention layers.
    """

    layer_vectors = []

    for key in attention_hidden_layers:
        layer_vectors.append(
            attention_hidden_layers[key][0].squeeze(0)
        )

    last_vectors = layer_vectors[-3:]

    avg_vector = np.mean(last_vectors, axis=0)

    return torch.tensor(
        avg_vector,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

# ============================================================
# FEATURE SELECTION FUNCTION
# ============================================================


def extract_hidden_feature():
    """
    Select feature extraction method based on parser argument.

    Available feature types:
        - mid3_fc
        - last3_fc
        - mid3_att
        - last3_att
    """

    if args.feature_type == "mid3_fc":
        return get_mid3_fc_average()

    elif args.feature_type == "last3_fc":
        return get_last3_fc_average()

    elif args.feature_type == "mid3_att":
        return get_mid3_attention_average()

    elif args.feature_type == "last3_att":
        return get_last3_attention_average()

    else:
        raise ValueError(f"Invalid feature type: {args.feature_type}")

def extract_label(text):
    """
    Extract valid label from generated text safely.
    """
    text = text.strip()

    valid_labels = {
        "ILDC": ["Accept", "Reject"],
        "ECHR": ["No-Violation", "Violation"]  
    }


    labels_to_check = sorted(valid_labels[args.dataset_name], key=len, reverse=True)

    for label in labels_to_check:
        # Corrected raw string word boundary
        pattern = rf"\b{re.escape(label)}\b"
        
        if re.search(pattern, text, re.IGNORECASE):
            return label

    return text

def reverse_prediction(label):
    """
    Reverse binary prediction.
    """

    reverse_map = {
        "ILDC": {
            "Accept": "Reject",
            "Reject": "Accept"
        },
        "ECHR": {
            "Violation": "No-Violation",
            "No-Violation": "Violation"
        }
    }

    return reverse_map[args.dataset_name].get(
        label,
        label
    )

# ============================================================
# GENERATE RESPONSE
# ============================================================


def generate_response(
    input_ids,
    model,
    tokenizer,
    stop_token,
    max_length=5
):
    generated_tokens = []

    hall_score = None

    for step in range(max_length):

        with torch.no_grad():

            logits = model(input_ids).logits

            next_token = logits[:, -1].argmax(dim=-1)

        # ====================================================
        # FIRST TOKEN HALLUCINATION CHECK
        # ====================================================

        if step == 0:

            with torch.no_grad():

                hidden_feature = extract_hidden_feature()

                feature_np = (
                    hidden_feature
                    .detach()
                    .cpu()
                    .numpy()
                )

                feature_np = (feature_np - mean) / std

                feature_tensor = torch.tensor(
                    feature_np,
                    dtype=torch.float32
                ).to(device)

                hall_logits = hallucination_clf(
                    feature_tensor
                )

                hall_score = (
                    hall_logits[0, 1]
                    - hall_logits[0, 0]
                ).item()

            # =================================================
            # REF MODE
            # =================================================
            # Early stop and return "Not Sure"
            # =================================================

            if (
                args.decision_mode == "REF"
                and hall_score < HALLUCINATION_THRESHOLD
            ):

                not_sure_ids = tokenizer(
                    "Not Sure",
                    return_tensors="pt"
                ).input_ids.to(model.device)[0]

                return (
                    not_sure_ids
                    .detach()
                    .cpu()
                    .numpy()
                )

        # ====================================================
        # APPEND GENERATED TOKEN
        # ====================================================

        input_ids = torch.cat(
            [input_ids, next_token.unsqueeze(0)],
            dim=1
        )

        generated_tokens.append(
            next_token.item()
        )

        # Stop if EOS token generated
        if next_token.item() == stop_token:
            break

    # ========================================================
    # DECODE FULL GENERATED RESPONSE
    # ========================================================

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # ========================================================
    # REV MODE
    # ========================================================
    # Reverse final label if hallucination detected
    # ========================================================

    if (
        args.decision_mode == "REV"
        and hall_score < HALLUCINATION_THRESHOLD
    ):

        predicted_label = extract_label(
            generated_text
        )

        reversed_label = reverse_prediction(
            predicted_label
        )

        reversed_ids = tokenizer(
            reversed_label,
            return_tensors="pt"
        ).input_ids.to(model.device)[0]

        return (
            reversed_ids
            .detach()
            .cpu()
            .numpy()
        )

    # ========================================================
    # NORMAL RESPONSE
    # ========================================================

    return np.array(generated_tokens)


# ============================================================
# ANSWER QUESTION
# ============================================================


def answer_question(question, model, tokenizer, stop_token):
    """
    Prepare prompt and generate response.
    """

    prompt = build_chat_prompt(question, tokenizer)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(model.device)

    input_ids = smart_truncate(input_ids)

    response = generate_response(
        input_ids=input_ids,
        model=model,
        tokenizer=tokenizer,
        stop_token=stop_token,
        max_length=args.max_new_tokens
    )

    return tokenizer.decode(response, skip_special_tokens=True)


# ============================================================
# MAIN FUNCTION
# ============================================================


def main():

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_dataset_from_csv()
    MODEL_DIR = Path("./.cache/models/")

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_REPOS[args.model_name][0]}/{args.model_name}",
        use_fast=True
    )

    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        f"{MODEL_REPOS[args.model_name][0]}/{args.model_name}",
        cache_dir=MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.eval()
    model.config.use_cache = False

    # --------------------------------------------------------
    # Register hooks
    # --------------------------------------------------------

    for name, module in model.named_modules():

        # Feed-forward hooks
        if re.match(MODEL_REPOS[args.model_name][1], name):
            module.register_forward_hook(
                partial(save_fc_hidden, name)
            )

        # Attention hooks
        if re.match(MODEL_REPOS[args.model_name][2], name):
            module.register_forward_hook(
                partial(save_attention_hidden, name)
            )

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    os.makedirs(config["output_dir"], exist_ok=True)

    output_file = os.path.join(
        config["output_dir"],
        f"results_{args.model_name}_{args.dataset_name}_{datetime.now().strftime('%m%d_%H%M')}_{args.feature_type}_{args.decision_mode}.json"
    )

    results = []

    # --------------------------------------------------------
    # Save checkpoint function
    # --------------------------------------------------------

    def save_checkpoint(sample_count):

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"Checkpoint saved after {sample_count} samples")

    # --------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------

    stop_token = tokenizer.eos_token_id

    with torch.no_grad():

        for idx, (question, answer) in enumerate(
            tqdm(dataset),
            start=1
        ):

            # Clear previous hidden states
            attention_hidden_layers.clear()
            fully_connected_hidden_layers.clear()

            response = answer_question(
                question,
                model,
                tokenizer,
                stop_token
            )

            results.append({
                "id": idx,
                "question": question,
                "gold_answer": answer,
                "model_response": response
            })

            # =================================================
            # CHECKPOINT SAVE EVERY 100 SAMPLES
            # =================================================

            if idx % args.save_every == 0:
                save_checkpoint(idx)

        # Final save
        save_checkpoint(len(results))

    print("Finished inference")
    print("Results saved at:", output_file)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

