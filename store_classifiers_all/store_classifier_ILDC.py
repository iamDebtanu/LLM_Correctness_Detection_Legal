import os
import pickle
import json
import numpy as np
import torch
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ----------------------------
# Config
# ----------------------------
gpu = "0"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 20
learning_rate = 1e-3
dropout_mlp = 0.1

# ----------------------------
# Seed
# ----------------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(123)

# ----------------------------
# Model
# ----------------------------
class FFClassifier(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Train function
# ----------------------------
def train_ff(X, y):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=123, stratify=y_tmp
    )

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_val   = (X_val - mean) / std
    X_test  = (X_test - mean) / std

    # X_train = torch.tensor(X_train, dtype=torch.float32)
    X_train = X_train.float().to(device)
    y_train = torch.tensor(y_train, dtype=torch.long)
    # y_train = y_train.long().to(device)

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=True)

    model = FFClassifier(X.shape[1], dropout=dropout_mlp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_roc = -1
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # logits = model(torch.tensor(X_val, device=device, dtype=torch.float32))
            logits = model(X_val.to(device))
            val_score = (logits[:,1] - logits[:,0]).cpu().numpy()

        val_roc = roc_auc_score(y_val, val_score)

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)

    with torch.no_grad():
        # logits = model(torch.tensor(X_test, device=device, dtype=torch.float32))
        logits = model(X_test.to(device))
        score = (logits[:,1] - logits[:,0]).cpu().numpy()

    preds = torch.argmax(logits, dim=1).cpu().numpy()

    return model, roc_auc_score(y_test, score), accuracy_score(y_test, preds), mean, std

# ----------------------------
# Load labels
# ----------------------------
with open("result_llm/results_ILDC_train/gemma-7b-it_ILDC_start1-0_end-2500_3_22.json", "r") as f:
    data = json.load(f)

correct = np.array([i["correct"] for i in data])

# ----------------------------
# Input pickle
# ----------------------------
results_file = "result_llm/results_ILDC_train/gemma-7b-it_ILDC_start1-0_end-2500_3_22.pickle"

with open(results_file, "rb") as f:
    results = pickle.load(f)

model_dir = "saved_model/saved_model_gemma_ILDC"
os.makedirs(model_dir, exist_ok=True)

out = {}

# ============================================================
# 1. FULL LAYER-WISE CLASSIFIERS
# ============================================================
print("\n=== Layer-wise classifiers ===")

fc_layers = results["first_fully_connected"][0].shape[0]
for layer in range(fc_layers):
    # X = np.stack([x[layer] for x in results["first_fully_connected"]])
    X = torch.stack([torch.tensor(x[layer], dtype=torch.float32) for x in results["first_fully_connected"]])

    model, roc, acc, m, s = train_ff(X, correct)

    out[f"fc_{layer}_roc"] = roc
    out[f"fc_{layer}_acc"] = acc

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "mean": m,
        "std": s,
    }, os.path.join(model_dir, f"fc_layer_{layer}.pth"))

att_layers = results["first_attention"][0].shape[0]
for layer in range(att_layers):
    # X = np.stack([x[layer] for x in results["first_attention"]])
    X = torch.stack([torch.tensor(x[layer], dtype=torch.float32) for x in results["first_attention"]])

    model, roc, acc, m, s = train_ff(X, correct)

    out[f"att_{layer}_roc"] = roc
    out[f"att_{layer}_acc"] = acc

    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "mean": m,
        "std": s,
    }, os.path.join(model_dir, f"att_layer_{layer}.pth"))

# ============================================================
# 2. MID-3 AVERAGE
# ============================================================
print("\n=== Mid-3 layers ===")

# mid_layers = [14, 15, 16]
mid_layers = [12, 13, 14]

# fc_mid = np.stack([
#     np.mean(x[mid_layers], axis=0)
#     for x in results["first_fully_connected"]
# ])

fc_mid = torch.stack([
    torch.stack([torch.tensor(x[i], dtype=torch.float32) for i in mid_layers]).mean(dim=0)
    for x in results["first_fully_connected"]])
model, roc, acc, m, s = train_ff(fc_mid, correct)
out["fc_mid3_roc"] = roc
out["fc_mid3_acc"] = acc

torch.save({
    "state_dict": model.state_dict(),
    "layers": mid_layers,
    "input_dim": fc_mid.shape[1],
    "mean": m,
    "std": s,
}, os.path.join(model_dir, "fc_mid3.pth"))

# att_mid = np.stack([
#     np.mean(x[mid_layers], axis=0)
#     for x in results["first_attention"]
# ])
att_mid = torch.stack([
    torch.stack([torch.tensor(x[i], dtype=torch.float32) for i in mid_layers]).mean(dim=0)
    for x in results["first_attention"]])
    
model, roc, acc, m, s = train_ff(att_mid, correct)
out["att_mid3_roc"] = roc
out["att_mid3_acc"] = acc

torch.save({
    "state_dict": model.state_dict(),
    "layers": mid_layers,
    "input_dim": att_mid.shape[1],
    "mean": m,
    "std": s,
}, os.path.join(model_dir, "att_mid3.pth"))

# ============================================================
# 3. LAST-3 AVERAGE
# ============================================================
print("\n=== Last-3 layers ===")

last_layers = [-3, -2, -1]

# fc_last = np.stack([
#     np.mean(x[last_layers], axis=0)
#     for x in results["first_fully_connected"]
# ])
fc_last = torch.stack([
    torch.stack([torch.tensor(x[i], dtype=torch.float32) for i in last_layers]).mean(dim=0)
    for x in results["first_fully_connected"]])


model, roc, acc, m, s = train_ff(fc_last, correct)
out["fc_last3_roc"] = roc
out["fc_last3_acc"] = acc

torch.save({
    "state_dict": model.state_dict(),
    "layers": last_layers,
    "input_dim": fc_last.shape[1],
    "mean": m,
    "std": s,
}, os.path.join(model_dir, "fc_last3.pth"))

# att_last = np.stack([
#     np.mean(x[last_layers], axis=0)
#     for x in results["first_attention"]
# ])
att_last = torch.stack([
    torch.stack([torch.tensor(x[i], dtype=torch.float32) for i in last_layers]).mean(dim=0)
    for x in results["first_attention"]])

model, roc, acc, m, s = train_ff(att_last, correct)
out["att_last3_roc"] = roc
out["att_last3_acc"] = acc

torch.save({
    "state_dict": model.state_dict(),
    "layers": last_layers,
    "input_dim": att_last.shape[1],
    "mean": m,
    "std": s,
}, os.path.join(model_dir, "att_last3.pth"))

# ============================================================
# SAVE RESULTS
# ============================================================
with open(os.path.join(model_dir, "final_results.txt"), "w") as f:
    for k, v in out.items():
        f.write(f"{k}: {v:.4f}\n")

print("\n✅ ALL DONE")