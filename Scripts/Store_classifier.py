# ============================================================
# Import Necessary Libraries
# ============================================================

import os
import json
import pickle
import random
import argparse

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser(
    description="Train classifiers on LLM hidden representations"
)

# Dataset selection
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    choices=["ECHR", "ILDC"],
    help="Choose dataset"
)

# Training mode
parser.add_argument(
    "--mode",
    type=str,
    default="all",
    choices=["all", "single", "multi", "average", "mid3", "last3"],
    help="Training mode"
)

# Single layer input
parser.add_argument(
    "--layer",
    type=int,
    default=None,
    help="Layer number for single layer training"
)

# Multiple layer input
parser.add_argument(
    "--layers",
    nargs="+",
    type=int,
    default=None,
    help="Multiple layer numbers"
)

# Representation type
parser.add_argument(
    "--representation",
    type=str,
    default="fc",
    choices=["fc", "att"],
    help="Choose feature type: fc or att"
)

args = parser.parse_args()


# ============================================================
# Device Configuration
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# Hyperparameters
# ============================================================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.1
PATIENCE = 5
SEED = 123


# ============================================================
# Dataset Configuration
# ============================================================

DATASET_CONFIG = {
    "ECHR": {
        "json_file": "results_llm/ECHR/Meta-Llama-3.1-8B-Instruct_ECHR_5_20.json", 
        "pickle_file": "results_llm/ECHR/Meta-Llama-3.1-8B-Instruct_ECHR_5_20.pickle",
        "save_dir": "saved_model/saved_model_meta_ECHR"
    },

    "ILDC": {
        "json_file": "results_llm/ILDC/Qwen2.5-7B-Instruct_ILDC_5_20.json",
        "pickle_file": "results_llm/ILDC/Qwen2.5-7B-Instruct_ILDC_5_20.pickle",
        "save_dir": "saved_model/saved_model_qwen_ILDC"
    }
}


# ============================================================
# Set Seed
# ============================================================


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Feed Forward Classifier (CD)
# ============================================================

class FFClassifier(torch.nn.Module):

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
# Normalize Features
# ============================================================


def normalize_data(X_train, X_val, X_test):

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


# ============================================================
# Train Classifier
# ============================================================


def train_classifier(X, y):

    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=SEED,
        stratify=y_temp
    )

    # Normalize features
    X_train, X_val, X_test, mean, std = normalize_data(
        X_train,
        X_val,
        X_test
    )

    # Convert to tensors
    X_train = X_train.float().to(DEVICE)
    X_val = X_val.float().to(DEVICE)
    X_test = X_test.float().to(DEVICE)

    y_train = torch.tensor(y_train, dtype=torch.long)

    # Create DataLoader
    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Create model
    model = FFClassifier(
        input_dim=X.shape[1],
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_roc = -1
    no_improvement = 0

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)

            loss = torch.nn.functional.cross_entropy(
                logits,
                yb
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val)
            val_scores = (
                val_logits[:, 1] - val_logits[:, 0]
            ).cpu().numpy()

        val_roc = roc_auc_score(y_val, val_scores)

        # print(
        #     f"Epoch {epoch+1} | Validation ROC-AUC: {val_roc:.4f}"
        # )

        # Save best model
        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_state = model.state_dict()
            no_improvement = 0

        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= PATIENCE:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Test evaluation
    model.eval()

    with torch.no_grad():
        test_logits = model(X_test)

        test_scores = (
            test_logits[:, 1] - test_logits[:, 0]
        ).cpu().numpy()

    predictions = torch.argmax(
        test_logits,
        dim=1
    ).cpu().numpy()

    roc = roc_auc_score(y_test, test_scores)
    acc = accuracy_score(y_test, predictions)

    return model, roc, acc, mean, std


# ============================================================
# Load Dataset
# ============================================================

config = DATASET_CONFIG[args.dataset_name]

print(f"Loading dataset: {args.dataset_name}")

# Load labels
with open(config["json_file"], "r") as f:
    data = json.load(f)

labels = np.array([
    item["correct"] for item in data
])

# Load representations
with open(config["pickle_file"], "rb") as f:
    results = pickle.load(f)


# ============================================================
# Choose Representation Type
# ============================================================

if args.representation == "fc":
    representations = results["first_fully_connected"]

else:
    representations = results["first_attention"]


# ============================================================
# Create Save Directory
# ============================================================

save_dir = os.path.join(
    config["save_dir"],
    f"{args.representation}_{args.mode}"
)

os.makedirs(save_dir, exist_ok=True)

# ============================================================
# Dictionary to Store Results
# ============================================================

out = {} 

# ============================================================
# Helper Function
# ============================================================


def extract_single_layer_features(data, layer_number):

    return torch.stack([
        torch.tensor(sample[layer_number], dtype=torch.float32)
        for sample in data
    ])

# ============================================================
# Helper Function: Get Mid-3 Layers
# ============================================================

def get_mid3_layers(total_layers):
    """
    Dynamically select middle 3 layers.

    Example:
    32 layers -> [14, 15, 16] (indices)
    28 layers -> [12, 13, 14]
    """

    mid = total_layers // 2

    return [mid - 2, mid -1, mid]

# ============================================================
# Mode 1: Train All Layers
# ============================================================

if args.mode == "all":

    total_layers = representations[0].shape[0]

    for layer in range(total_layers):

        print(f"\\nTraining Layer {layer}")

        X = extract_single_layer_features(
            representations,
            layer
        )

        model, roc, acc, mean, std = train_classifier(
            X,
            labels
        )

        # print(f"ROC-AUC: {roc:.4f}")
        # print(f"Accuracy: {acc:.4f}")

        # Store metrics
        out[f"layer_{layer}_roc"] = roc
        out[f"layer_{layer}_acc"] = acc

        # Save model
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "mean": mean,
            "std": std,
            "layer": layer
        }, os.path.join(save_dir, f"layer_{layer}.pth"))


# ============================================================
# Mode 2: Train Single Layer
# ============================================================

elif args.mode == "single":

    if args.layer is None:
        raise ValueError(
            "Please provide --layer for single mode"
        )

    print(f"Training single layer: {args.layer}")

    X = extract_single_layer_features(
        representations,
        args.layer
    )

    model, roc, acc, mean, std = train_classifier(
        X,
        labels
    )

    # print(f"ROC-AUC: {roc:.4f}")
    # print(f"Accuracy: {acc:.4f}")

    # Store metrics
    out[f"single_layer_{args.layer}_roc"] = roc
    out[f"single_layer_{args.layer}_acc"] = acc

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "mean": mean,
        "std": std,
        "layer": args.layer
    }, os.path.join(
        save_dir,
        f"single_layer_{args.layer}.pth"
    ))


# ============================================================
# Mode 3: Train Multiple Layers Separately
# ============================================================

elif args.mode == "multi":

    if args.layers is None:
        raise ValueError(
            "Please provide --layers for multi mode"
        )

    for layer in args.layers:

        print(f"\\nTraining Layer {layer}")

        X = extract_single_layer_features(
            representations,
            layer
        )

        model, roc, acc, mean, std = train_classifier(
            X,
            labels
        )

        # print(f"ROC-AUC: {roc:.4f}")
        # print(f"Accuracy: {acc:.4f}")

        # Store metrics
        out[f"layer_{layer}_roc"] = roc
        out[f"layer_{layer}_acc"] = acc

        # Save model
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "mean": mean,
            "std": std,
            "layer": layer
        }, os.path.join(
            save_dir,
            f"layer_{layer}.pth"
        ))


# ============================================================
# Mode 4: Average Multiple Layers
# ============================================================

elif args.mode == "average":

    if args.layers is None:
        raise ValueError(
            "Please provide --layers for average mode"
        )

    print(f"Averaging layers: {args.layers}")

    X = torch.stack([
        torch.stack([
            torch.tensor(sample[layer], dtype=torch.float32)
            for layer in args.layers
        ]).mean(dim=0)
        for sample in representations
    ])

    model, roc, acc, mean, std = train_classifier(
        X,
        labels
    )

    # print(f"ROC-AUC: {roc:.4f}")
    # print(f"Accuracy: {acc:.4f}")

    # Create layer name
    layer_name = "_".join(map(str, args.layers))

    # Store metrics
    out[f"avg_{layer_name}_roc"] = roc
    out[f"avg_{layer_name}_acc"] = acc

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "mean": mean,
        "std": std,
        "layers": args.layers
    }, os.path.join(
        save_dir,
        f"average_layers_{layer_name}.pth"
    ))

# ============================================================
# Mode 5: Mid-3 Layers
# ============================================================

elif args.mode == "mid3":

    total_layers = representations[0].shape[0]

    # Dynamically get middle layers
    mid_layers = get_mid3_layers(total_layers)

    print(f"Total Layers: {total_layers}")
    print(f"Using Mid-3 layers: {mid_layers}")

    X = torch.stack([
        torch.stack([
            torch.tensor(sample[layer], dtype=torch.float32)
            for layer in mid_layers
        ]).mean(dim=0)
        for sample in representations
    ])

    model, roc, acc, mean, std = train_classifier(
        X,
        labels
    )

    # print(f"ROC-AUC: {roc:.4f}")
    # print(f"Accuracy: {acc:.4f}")

    # Store metrics
    out["mid3_roc"] = roc
    out["mid3_acc"] = acc

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "mean": mean,
        "std": std,
        "layers": mid_layers
    }, os.path.join(
        save_dir,
        "mid3_layers.pth"
    ))
    
# ============================================================
# Mode 6: Last-3 Layers
# ============================================================

elif args.mode == "last3":

    # Last three layers
    last_layers = [-3, -2, -1]

    print(f"Averaging Last-3 layers: {last_layers}")

    X = torch.stack([
        torch.stack([
            torch.tensor(sample[layer], dtype=torch.float32)
            for layer in last_layers
        ]).mean(dim=0)
        for sample in representations
    ])

    model, roc, acc, mean, std = train_classifier(
        X,
        labels
    )

    # print(f"ROC-AUC: {roc:.4f}")
    # print(f"Accuracy: {acc:.4f}")

    # Store metrics
    out["last3_roc"] = roc
    out["last3_acc"] = acc

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "mean": mean,
        "std": std,
        "layers": last_layers
    }, os.path.join(
        save_dir,
        "last3_layers.pth"
    ))
# ============================================================
# Save Final Results
# ============================================================

results_file = os.path.join(
    save_dir,
    "final_results.txt"
)

with open(results_file, "w") as f:

    for k, v in out.items():
        f.write(f"{k}: {v:.4f}\n")


# ============================================================
# Training Complete
# ============================================================

print("\\nTraining completed successfully!")
print(f"Models saved in: {save_dir}")
print(f"Results saved in: {results_file}")

