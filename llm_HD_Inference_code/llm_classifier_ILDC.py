import json
import argparse
import functools
from datetime import datetime
from typing import Any, Dict
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict, Counter
from functools import partial
import re
# from captum.attr import IntegratedGradients
from string import Template
import pandas as pd
import os

# from huggingface_hub import login
#
# login("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")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class FFClassifier(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
HALLUCINATION_THRESHOLD = 0.0
HALL_CLF_PATH = "saved_model/saved_model_gemma_ILDC/att_last3.pth"
# HALL_CLF_PATH = "/home/rs/25CS91R16/saved_models/ff_att_first_layer12.pth"
# HIDDEN_SIZE = 14336   #4096 
# ckpt = torch.load(HALL_CLF_PATH, map_location=device)
ckpt = torch.load(HALL_CLF_PATH, map_location=device, weights_only=False)

hallucination_clf = FFClassifier(ckpt["input_dim"]).to(device)
hallucination_clf.load_state_dict(ckpt["state_dict"])
hallucination_clf.eval()
mean = ckpt["mean"]        # shape (1, D)
std = ckpt["std"]  
mean = mean.detach().cpu().numpy()
std = std.detach().cpu().numpy()
# ------------------ ARGUMENT PARSER ------------------
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B-Instruct",
                    help="Model key name from model_repos dictionary")

# parser.add_argument("--dataset_name", type=str, default="India_legal_qa",
                    # help="Dataset to use: India_legal_qa, US_constitution_legal_qa, US_copyright_legal_qa")

parser.add_argument("--iteration", type=int, default=0,
                    help="Chunk index for parallelization")

parser.add_argument("--interval", type=int, default=2500,
                    help="How many samples per iteration chunk")

args = parser.parse_args()


# ------------------ DATA PARAMS ------------------
iteration = args.iteration
interval = args.interval
start = iteration * interval
end = start + interval
# dataset_name = args.dataset_name
dataset_name = "ILDC"
model_name = args.model_name

def build_chat_prompt(question, tokenizer):
    if "gemma" in model_name.lower():
        system_instruction = (
            "You are a legal expert specialized in Indian law. "
            "Your task is to analyze the given case proceeding strictly based on "
            "principles of Indian law and established judicial reasoning, and determine "
            "whether the appeal should be accepted or rejected. "
            "Your decision must be solely based on the facts and legal issues presented "
            "in the case. "
            "Output only one word: either \"Accept\" or \"Reject\"."
        )
        chat = [
            {
                "role": "user",
                "content": f"{system_instruction}\n\n{question}"
            }
        ]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal expert specialized in Indian law. "
                    "Your task is to analyze the given case proceeding strictly based on "
                    "principles of Indian law and established judicial reasoning, and determine "
                    "whether the appeal should be accepted or rejected. "
                    "Your decision must be solely based on the facts and legal issues presented "
                    "in the case. "
                    "Output only one word: either \"Accept\" or \"Reject\"."
                )
            },
            {
                "role": "user",
                "content": f"{question}"
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    return prompt


# IO
data_dir = Path(".")
model_dir = Path("./.cache/models/")
# results_dir = Path("./result_llm/results_ILDC_LLM_HD/")

# Hardware
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# Integrated Grads
# ig_steps = 64
# internal_batch_size = 4

# Model config
model_num_layers = {
    "falcon-40b": 60,
    "falcon-7b": 32,
    "falcon-7b-instruct": 32,
    "mistral-7b-instruct-v0.3": 32,
    "gemma-7b-it": 28,
    "open_llama_13b": 40,
    "open_llama_7b": 32,
    "Meta-Llama-3.1-8B-Instruct": 32,
    "opt-6.7b": 32,
    "opt-30b": 48,
}

layer_number = -1
assert layer_number < model_num_layers[model_name]

coll_str = "[0-9]+" if layer_number == -1 else str(layer_number)

model_repos = {
    "falcon-40b": ("tiiuae",
                   f".*transformer.h.{coll_str}.mlp.dense_4h_to_h",
                   f".*transformer.h.{coll_str}.self_attention.dense"),

    "falcon-7b": ("tiiuae",
                  f".*transformer.h.{coll_str}.mlp.dense_4h_to_h",
                  f".*transformer.h.{coll_str}.self_attention.dense"),

    "falcon-7b-instruct": ("tiiuae",
                  f".*transformer.h.{coll_str}.mlp.dense_4h_to_h",
                  f".*transformer.h.{coll_str}.self_attention.dense"),

    "mistral-7b-instruct-v0.3": ("mistralai",
                  f".*model.layers.{coll_str}.mlp.up_proj",
                  f".*model.layers.{coll_str}.self_attn.o_proj"),

    "gemma-7b-it": ("google",
                  f".*model.layers.{coll_str}.mlp.up_proj",
                  f".*model.layers.{coll_str}.self_attn.o_proj"),

    "Meta-Llama-3.1-8B-Instruct": ("meta-llama",
                    f".*model.layers.{coll_str}.mlp.up_proj",
                    f".*model.layers.{coll_str}.self_attn.o_proj"),


    "open_llama_13b": ("openlm-research",
                       f".*model.layers.{coll_str}.mlp.up_proj",
                       f".*model.layers.{coll_str}.self_attn.o_proj"),

    "open_llama_7b": ("openlm-research",
                      f".*model.layers.{coll_str}.mlp.up_proj",
                      f".*model.layers.{coll_str}.self_attn.o_proj"),

    "opt-6.7b": ("facebook",
                 f".*model.decoder.layers.{coll_str}.fc2",
                 f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),

    "opt-30b": ("facebook",
                f".*model.decoder.layers.{coll_str}.fc2",
                f".*model.decoder.layers.{coll_str}.self_attn.out_proj"),
}


# ------------------ STORAGE ------------------
fully_connected_hidden_layers = defaultdict(list)
attention_hidden_layers = defaultdict(list)
attention_forward_handles = {}
fully_connected_forward_handles = {}


# ------------------ HOOK FUNCTIONS ------------------
# def save_fully_connected_hidden(layer_name, mod, inp, out):
#     fully_connected_hidden_layers[layer_name].append(
#         out.squeeze().detach().to(torch.float32).cpu().numpy()
#     )
#
#
# def save_attention_hidden(layer_name, mod, inp, out):
#     attention_hidden_layers[layer_name].append(
#         out.squeeze().detach().to(torch.float32).cpu().numpy()
#     )
def save_fully_connected_hidden(layer_name, mod, inp, out):
    # Only keep last token activation
    last_token = out[:, -1, :].detach().to(torch.float16).cpu().numpy()
    fully_connected_hidden_layers[layer_name].append(last_token)


def save_attention_hidden(layer_name, mod, inp, out):
    last_token = out[:, -1, :].detach().to(torch.float16).cpu().numpy()
    attention_hidden_layers[layer_name].append(last_token)




def load_data_from_csv(
        csv_path,
        start=0,
        end=None
    ):
        df = pd.read_csv(csv_path)

        end = len(df)

        data = []
        for i in range(start, min(end, len(df))):
            ex = df.iloc[i]

            user_content = (
                "Predict the outcome of the appeal in the following case proceeding "
                "based strictly on Indian law.\n\n"
                f"Case Proceeding:\n{ex['text']}\n\n"
                "Decision:"
            )

            label_map = {
                1: "Accept",
                0: "Reject"
            }

            answer = label_map[ex["label"]]


            # Keep same format as before (list)
            # answer = [ex["answer"]]

            data.append((user_content, answer))

        return data





def get_next_token(x, model):
    with torch.no_grad():
        return model(x).logits

def get_mid3_avg_token_fc(layer_start, layer_end):
    layer_name = model_repos[model_name][1][2:].split(coll_str)
    fc = [
        fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][0].squeeze(0)
        for i in range(layer_start, layer_end)
    ]
    # return torch.tensor(np.mean([fc[14], fc[15], fc[16]], axis=0),dtype=torch.float32).unsqueeze(0).to(device)
    return torch.tensor(np.mean([fc[12], fc[13], fc[14]], axis=0),dtype=torch.float32).unsqueeze(0).to(device)

def get_last3_avg_token_fc(layer_start, layer_end):
    layer_name = model_repos[model_name][1][2:].split(coll_str)
    fc = [
        fully_connected_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][0].squeeze(0)
        for i in range(layer_start, layer_end)
    ]
    return torch.tensor(np.mean([fc[-3], fc[-2], fc[-1]], axis=0),dtype=torch.float32).unsqueeze(0).to(device)

def get_mid3_avg_token_att( layer_start, layer_end):
    layer_name = model_repos[model_name][2][2:].split(coll_str)
    fc = [
        attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][0].squeeze(0)
        for i in range(layer_start, layer_end)
    ]
    # return torch.tensor(np.mean([fc[14], fc[15], fc[16]], axis=0),dtype=torch.float32).unsqueeze(0).to(device)
    return torch.tensor(np.mean([fc[12], fc[13], fc[14]], axis=0),dtype=torch.float32).unsqueeze(0).to(device)

def get_last3_avg_token_att(layer_start, layer_end):
    layer_name = model_repos[model_name][2][2:].split(coll_str)
    fc = [
        attention_hidden_layers[f'{layer_name[0]}{i}{layer_name[1]}'][0].squeeze(0)
        for i in range(layer_start, layer_end)
    ]
    return torch.tensor(np.mean([fc[-3], fc[-2], fc[-1]], axis=0),dtype=torch.float32).unsqueeze(0).to(device)


def generate_response(x, model, tokenizer, stop_token = None, *, max_length=10):
    response = []
    layer_start, layer_end = get_start_end_layer(model)

    for step in range(max_length):


        with torch.no_grad():
            logits = model(x).logits
            next_token = logits[:, -1].argmax(dim=-1)

        # ================= FIRST-TOKEN GATING =================
        if step == 0:
            with torch.no_grad():
                first_fc = get_last3_avg_token_att(layer_start, layer_end)
                y = first_fc.detach().cpu().numpy()
                y = (y - mean) / std
                y = torch.tensor(y, dtype=torch.float32).to(device)

                hall_logits = hallucination_clf(y)
                hall_score = (hall_logits[0, 1] - hall_logits[0, 0]).item()

            if hall_score < HALLUCINATION_THRESHOLD:
                not_sure_ids = tokenizer("Not Sure", return_tensors="pt").input_ids.to(model.device)[0]
                del logits
                torch.cuda.empty_cache()

                return not_sure_ids.detach().cpu().numpy() #, logits.squeeze()
        # =====================================================

        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        response.append(next_token.item())
        del logits
        torch.cuda.empty_cache()

        if next_token.item() == stop_token:
            break

    return np.array(response)


def smart_truncate(input_ids, max_tokens=8192):
    seq_len = input_ids.shape[1]

    if seq_len <= max_tokens:
        return input_ids

    keep_head = max_tokens // 2
    keep_tail = max_tokens - keep_head

    head = input_ids[:, :keep_head]
    tail = input_ids[:, -keep_tail:]

    return torch.cat([head, tail], dim=1)

def answer_question(question, model, tokenizer,stop_token = None, *, max_length=5):
    prompt = build_chat_prompt(question, tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # print("Prompt token length:", input_ids.shape[1])
    input_ids = smart_truncate(input_ids, 8192)

    response = generate_response(input_ids, model,tokenizer, max_length=max_length,stop_token=stop_token)
    return response, input_ids.shape[-1]




def answer_qa(question, targets, model, tokenizer,stop_token=None):
    response, start_pos = answer_question(question, model, tokenizer,stop_token)
    str_response = tokenizer.decode(response, skip_special_tokens=True)
    # if "gemma" in model_name.lower(): 
    match_pred = re.search(r'[A-Za-z-]+', str_response)
    pred = match_pred.group(0).lower() if match_pred else ""
    # print(pred)
    # print(targets.lower())
    #

    # match = re.search(r'\b([ABCD])\b', str_response)
    # pred = match.group(1).lower() if match else None
    correct = pred == targets.lower()
    # else:
    #     clean = str_response.strip().split(',')[0].split('.')[0].lower()
    #     correct = clean == targets.lower()
        # print(targets.lower())
        # print(clean)

    return str_response


def get_start_end_layer(model):
   
    if "llama" in model_name.lower() or "mistral" in model_name.lower() or "gemma" in model_name.lower():
        layer_count = model.model.layers

    elif "falcon" in model_name.lower():
        layer_count = model.transformer.h

    else:
        # OPT / BART / others
        layer_count = model.model.decoder.layers

    layer_st = 0 if layer_number == -1 else layer_number
    layer_en = len(layer_count) if layer_number == -1 else layer_number + 1
    return layer_st, layer_en



def compute_and_save_results():

    dataset = load_data_from_csv("Datasets/ILDC_test.csv")
    question_asker = answer_qa

    # Model
    model_loader = AutoModelForCausalLM
    token_loader = AutoTokenizer

    tokenizer = token_loader.from_pretrained(
        f'{model_repos[model_name][0]}/{model_name}',
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token_id

    model = model_loader.from_pretrained(
        f'{model_repos[model_name][0]}/{model_name}',
        cache_dir=model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.eval()

    # Hooks
    for name, module in model.named_modules():
        if re.match(f'{model_repos[model_name][1]}$', name):
            fully_connected_forward_handles[name] = module.register_forward_hook(
                partial(save_fully_connected_hidden, name))

        if re.match(f'{model_repos[model_name][2]}$', name):
            attention_forward_handles[name] = module.register_forward_hook(
                partial(save_attention_hidden, name))

    # -------------------- RESULTS --------------------
    json_results = []

    # -------------------- SAVE PATH --------------------
    out_dir = "results_llm_HD/ILDC"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"gated_results_{datetime.now().strftime('%m%d_%H%M')}.json"
    )

    SAVE_EVERY = 100

    def save_checkpoint(n):
        with open(out_file, "w", encoding="utf-8") as f:
            # json.dump(json_results, f, indent=2, ensure_ascii=False)
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        print(f"💾 Checkpoint saved at {n} samples")

    # -------------------- MAIN LOOP --------------------
    with torch.no_grad():
        for idx, (q, a) in enumerate(tqdm(dataset), start=1):
            fully_connected_hidden_layers.clear()
            attention_hidden_layers.clear()

            response = question_asker(
                q,a, model, tokenizer, stop_token
            )

            json_results.append({
                "id": idx,
                "question": q,
                "answer": a,
                "response": response
            })

            # 🔥 CHECKPOINT SAVE
            if idx % SAVE_EVERY == 0:
                save_checkpoint(idx)

        # -------------------- FINAL SAVE --------------------
        save_checkpoint(len(json_results))
        print("✅ Final saved:", out_file)


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    compute_and_save_results()



