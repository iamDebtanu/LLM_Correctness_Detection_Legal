import os
import gc
import re
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# GPU CONFIGURATION
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================================================
# ARGUMENT PARSER
# ============================================================

parser = argparse.ArgumentParser(
    description="Unified LLM inference script for ILDC and ECHR datasets"
)

parser.add_argument(
    "--dataset_name",
    type=str,
    choices=["ILDC", "ECHR"],
    default="ILDC",
    help="Dataset to run inference on"
)

parser.add_argument(
    "--model_name",
    type=str,
    default="Meta-Llama-3.1-8B-Instruct",
    help="Model name from model_repos"
)

parser.add_argument(
    "--iteration",
    type=int,
    default=0,
    help="Chunk index for parallel execution"
)

parser.add_argument(
    "--interval",
    type=int,
    default=2500,
    help="Number of samples per chunk"
)

args = parser.parse_args()


# ============================================================
# BASIC SETTINGS
# ============================================================

DATASET_NAME = args.dataset_name
MODEL_NAME = args.model_name

START = args.iteration * args.interval
END = START + args.interval

MODEL_DIR = Path("./.cache/models/")
RESULTS_DIR = Path(f"results_llm/{DATASET_NAME}/")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET CONFIGURATION
# ============================================================
# All dataset-specific settings are stored here.
# ============================================================

DATASET_CONFIG = {
    "ILDC": {
        "csv_path": "Datasets/ILDC_train.csv", ##ILDC dataset path 
        "text_column": "text",
        "label_column": "label",
        "label_map": {
            1: "Accept",
            0: "Reject"
        },
        "system_prompt": (
                    "You are a legal expert specialized in Indian law. "
                    "Your task is to analyze the given case proceeding strictly based on "
                    "principles of Indian law and established judicial reasoning, and determine "
                    "whether the appeal should be accepted or rejected. "
                    "Your decision must be solely based on the facts and legal issues presented "
                    "in the case. "
                    "Output only one word: either \"Accept\" or \"Reject\"."
                ),
        "user_prompt_template": (
            "Predict the outcome of the appeal in the following "
            "case proceeding based strictly on Indian law.\n\n"
            "Case Proceeding:\n{text}\n\n"
            "Decision:"
        )
    },

    "ECHR": {
        "csv_path": "Datasets/test_ECHR_Dataset_binary.csv", ##ECHR dataset path
        "text_column": "text",
        "label_column": "binary_judgement",
        "label_map": {
            1: "Violation",
            0: "No-Violation"
        },
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
        "user_prompt_template": (
            "Based on the European Convention on Human Rights, "
            "determine whether the following case proceeding "
            "shows a violation of a human rights article.\n\n"
            "Case Proceeding:\n{text}\n\n"
            "Decision:"
        )
    }
}

CONFIG = DATASET_CONFIG[DATASET_NAME]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NUM_LAYERS = {
    "Meta-Llama-3.1-8B-Instruct": 32,
    "mistral-7b-instruct-v0.3": 32,
    "gemma-7b-it": 28,
    "Qwen2.5-7B-Instruct": 28,
}

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
# STORAGE FOR ACTIVATIONS
# ============================================================

fully_connected_hidden_layers = defaultdict(list)
attention_hidden_layers = defaultdict(list)


# ============================================================
# HOOK FUNCTIONS
# ============================================================
# These hooks store intermediate activations from:
# 1. MLP layers
# 2. Attention output layers
#
# Store only the hidden state of the final generated token to reduce memory.
# ============================================================


def save_fully_connected_hidden(layer_name, mod, inp, out):
    last_token = out[:, -1, :].detach().to(torch.float16).cpu().numpy()
    fully_connected_hidden_layers[layer_name].append(last_token)



def save_attention_hidden(layer_name, mod, inp, out):
    last_token = out[:, -1, :].detach().to(torch.float16).cpu().numpy()
    attention_hidden_layers[layer_name].append(last_token)


# ============================================================
# LOAD DATASET
# ============================================================


def load_dataset_data():
    """
    Loads dataset and converts each sample into:

    (question_prompt, gold_answer)
    """

    df = pd.read_csv(CONFIG["csv_path"])

    data = []

    for _, row in df.iterrows():

        question = CONFIG["user_prompt_template"].format(
            text=row[CONFIG["text_column"]]
        )

        answer = CONFIG["label_map"][row[CONFIG["label_column"]]]

        data.append((question, answer))

    return data[:20]


# ============================================================
# BUILD CHAT PROMPT
# ============================================================


def build_chat_prompt(question, tokenizer):
    """
    Creates final chat-formatted prompt.
    """

    messages = [
        {
            "role": "system",
            "content": CONFIG["system_prompt"]
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
# SMART TRUNCATION
# ============================================================
# Keeps both beginning and ending of long sequences.
# This is useful for long legal documents.
# ============================================================


def smart_truncate(input_ids, max_tokens=8192):

    if input_ids.shape[1] <= max_tokens:
        return input_ids

    keep_head = max_tokens // 2
    keep_tail = max_tokens - keep_head

    head = input_ids[:, :keep_head]
    tail = input_ids[:, -keep_tail:]

    return torch.cat([head, tail], dim=1)


# ============================================================
# RESPONSE GENERATION
# ============================================================


def generate_response(input_ids, model, stop_token, max_length=5):
    """
    Greedy decoding generation.
    """

    response = []

    with torch.no_grad():

        outputs = model(
            input_ids,
            use_cache=True,
            num_logits_to_keep=1
        )

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :].float().cpu(),
            dim=-1
        )

        for _ in range(max_length):

            token_id = next_token.item()
            response.append(token_id)

            if token_id == stop_token:
                break

            next_token_gpu = torch.tensor(
                [[token_id]],
                device=input_ids.device
            )

            outputs = model(
                next_token_gpu,
                use_cache=True,
                past_key_values=past_key_values,
                num_logits_to_keep=1
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(
                outputs.logits[:, -1, :].float().cpu(),
                dim=-1
            )

    return response


# ============================================================
# QUESTION ANSWERING
# ============================================================
# Decodes up to 5 generated tokens, extracts
# predicted label tokens from the response text, and
# compares them with the ground-truth label.
# ----------------------------------------------------


def answer_question(question, model, tokenizer, stop_token):

    prompt = build_chat_prompt(question, tokenizer)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(model.device)

    input_ids = smart_truncate(input_ids)

    response = generate_response(
        input_ids,
        model,
        stop_token
    )

    return response, input_ids.shape[-1]


# ============================================================
# PREDICTION EVALUATION
# ============================================================


def answer_qa(question, target, model, tokenizer, stop_token):

    response, start_pos = answer_question(
        question,
        model,
        tokenizer,
        stop_token
    )

    decoded_response = tokenizer.decode(
        response,
        skip_special_tokens=True
    )

    # Extract all alphabetic tokens
    matches = re.findall(r"[A-Za-z-]+", decoded_response)

    pred_tokens = [m.lower() for m in matches]

    # Check whether target label appears anywhere
    correct = target.lower() in pred_tokens

    return response, decoded_response, start_pos, correct


# ============================================================
# COLLECT ACTIVATIONS
# ============================================================


def collect_activations(storage_dict):

    activations = []

    for key in storage_dict:
        activations.append(storage_dict[key][0].squeeze(0))

    return np.stack(activations)


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_inference():

    dataset = load_dataset_data()

    print(f"Loaded {len(dataset)} samples")

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        f"{MODEL_REPOS[MODEL_NAME][0]}/{MODEL_NAME}",
        use_fast=True
    )

    tokenizer.pad_token = tokenizer.eos_token


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        f"{MODEL_REPOS[MODEL_NAME][0]}/{MODEL_NAME}",
        cache_dir=MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.eval()
    model.config.use_cache = True


    # --------------------------------------------------------
    # REGISTER HOOKS
    # --------------------------------------------------------

    for name, module in model.named_modules():

        if re.match(MODEL_REPOS[MODEL_NAME][1], name):
            module.register_forward_hook(
                partial(save_fully_connected_hidden, name)
            )

        if re.match(MODEL_REPOS[MODEL_NAME][2], name):
            module.register_forward_hook(
                partial(save_attention_hidden, name)
            )


    # --------------------------------------------------------
    # OUTPUT STORAGE
    # --------------------------------------------------------

    results_json = []

    results_pickle = {
        "response": [],
        "start_pos": [],
        "first_fully_connected": [],
        "first_attention": []
    }


    # --------------------------------------------------------
    # SAVE PATHS
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = f"{datetime.now().month}_{datetime.now().day}"

    json_path = RESULTS_DIR / (
        f"{MODEL_NAME}_{DATASET_NAME}_{timestamp}.json"
    )

    pickle_path = RESULTS_DIR / (
        f"{MODEL_NAME}_{DATASET_NAME}_{timestamp}.pickle"
    )


    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for idx, (question, answer) in enumerate(tqdm(dataset)):

        torch.cuda.empty_cache()
        gc.collect()

        fully_connected_hidden_layers.clear()
        attention_hidden_layers.clear()

        stop_token = tokenizer.eos_token_id

        response, decoded_response, start_pos, correct = answer_qa(
            question,
            answer,
            model,
            tokenizer,
            stop_token
        )


        # ----------------------------------------------------
        # STORE ACTIVATIONS
        # ----------------------------------------------------

        fc_activations = collect_activations(
            fully_connected_hidden_layers
        )

        att_activations = collect_activations(
            attention_hidden_layers
        )


        # ----------------------------------------------------
        # SAVE JSON RESULTS
        # ----------------------------------------------------

        results_json.append({
            "id": idx,
            "question": question,
            "gold_answer": answer,
            "prediction": decoded_response,
            "correct": bool(correct)
        })


        # ----------------------------------------------------
        # SAVE PICKLE FEATURES
        # ----------------------------------------------------

        results_pickle["response"].append(response)
        results_pickle["start_pos"].append(start_pos)
        results_pickle["first_fully_connected"].append(fc_activations)
        results_pickle["first_attention"].append(att_activations)


        # ----------------------------------------------------
        # CHECKPOINT SAVE FOR EVERY 100 
        # ----------------------------------------------------
        # Save intermediate results after every 100 samples to
        # prevent data loss during long inference runs.
        # ----------------------------------------------------

        if (idx + 1) % 100 == 0:

            with open(json_path, "w") as f:
                json.dump(results_json, f, indent=2)

            with open(pickle_path, "wb") as f:
                pickle.dump(results_pickle, f)

            print(f"Checkpoint saved at sample {idx + 1}")


    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    with open(pickle_path, "wb") as f:
        pickle.dump(results_pickle, f)


    print("Inference completed successfully")
    print(f"JSON saved to: {json_path}")
    print(f"Pickle saved to: {pickle_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_inference()
