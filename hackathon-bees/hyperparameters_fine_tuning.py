import gc
import os
import random
from collections import defaultdict

import numpy as np
import optuna
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

DATA_DIR   = "data"
DEVICE     = torch.device("mps")   # change to "cuda" or "cpu" if needed
BATCH_SIZE = 32
SEED       = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ==============================================================================
# DATA LOADING
# ==============================================================================

data_train  = pd.read_csv("data/train.csv")
all_paths   = [os.path.join(DATA_DIR, p) for p in data_train["id"]]
all_labels  = data_train["label"].values
num_classes = len(np.unique(all_labels))

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels,
    test_size=0.2,
    random_state=SEED,
    stratify=all_labels,
)

# ==============================================================================
# TRANSFORMS
# ==============================================================================

base_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

aug_transform = v2.Compose([
    v2.RandomRotation(30),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ==============================================================================
# DATASET
# ==============================================================================

class BeeDataset(Dataset):
    """
    Dataset for bee image classification with optional augmentation.

    Args:
        paths              : list of image file paths
        labels             : corresponding integer labels
        augment_factors    : dict mapping class_id -> repeat factor
        apply_augmentation : whether to apply data augmentation
        max_per_class      : cap on samples per class after augmentation
        image_cache        : if True, preload all images into RAM
    """

    def __init__(
        self,
        paths: list,
        labels: list,
        augment_factors: dict = None,
        apply_augmentation: bool = True,
        max_per_class: int = 300,
        image_cache: bool = True,
    ):
        self.original_labels    = labels
        self.augment_factors    = augment_factors or {}
        self.apply_augmentation = apply_augmentation
        self.max_per_class      = max_per_class
        self.image_cache        = image_cache

        if image_cache:
            print("Preloading images into RAM...")
            self.images_cache = [
                Image.open(p).convert("RGB") for p in tqdm(paths, desc="Cache")
            ]
        else:
            self.paths = paths

        self.indices, self.is_augmented = self._build_index()
        print(f"Dataset: {len(paths)} original images -> {len(self.indices)} samples")

    def _build_index(self):
        """Build sample index with augmentation and optional per-class capping."""
        indices, is_augmented = [], []

        if self.apply_augmentation and self.augment_factors:
            for i, label in enumerate(self.original_labels):
                factor = self.augment_factors.get(label, 1)
                for k in range(factor):
                    indices.append(i)
                    is_augmented.append(k > 0)
        else:
            return list(range(len(self.original_labels))), [False] * len(self.original_labels)

        # Cap samples per class
        if self.max_per_class is not None:
            class_positions = defaultdict(list)
            for pos, idx in enumerate(indices):
                class_positions[self.original_labels[idx]].append(pos)

            keep = []
            for positions in class_positions.values():
                keep.extend(
                    random.sample(positions, self.max_per_class)
                    if len(positions) > self.max_per_class
                    else positions
                )
            keep.sort()
            indices      = [indices[p]      for p in keep]
            is_augmented = [is_augmented[p] for p in keep]

        combined = list(zip(indices, is_augmented))
        random.shuffle(combined)
        indices, is_augmented = zip(*combined) if combined else ([], [])
        return list(indices), list(is_augmented)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img = (
            self.images_cache[original_idx].copy()
            if self.image_cache
            else Image.open(self.paths[original_idx]).convert("RGB")
        )
        label     = self.original_labels[original_idx]
        transform = aug_transform if self.is_augmented[idx] else base_transform
        return transform(img), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# METRICS
# ==============================================================================

def compute_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return f1_score(
        labels.numpy(), torch.argmax(preds, dim=1).numpy(),
        average="macro", zero_division=0,
    )

def compute_precision(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return precision_score(
        labels.numpy(), torch.argmax(preds, dim=1).numpy(),
        average="macro", zero_division=0,
    )

def compute_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return recall_score(
        labels.numpy(), torch.argmax(preds, dim=1).numpy(),
        average="macro", zero_division=0,
    )

# ==============================================================================
# OPTUNA OBJECTIVE
# ==============================================================================

def objective(trial: optuna.Trial) -> float:

    # --- Hyperparameters ---
    divisor       = trial.suggest_int(  "divisor",       5,   50)
    max_per_class = trial.suggest_int(  "max_per_class", 50,  500, step=50)
    lr            = trial.suggest_float("lr",            1e-5, 1e-3, log=True)
    dropout       = trial.suggest_float("dropout",       0.2,  0.6)
    num_epochs    = trial.suggest_int(  "num_epochs",    10,   30)

    # --- Augmentation factors (balance minority classes) ---
    class_counts    = np.bincount(train_labels)
    max_count       = class_counts.max() / divisor
    augment_factors = {
        c: int(np.ceil(max_count / cnt))
        for c, cnt in enumerate(class_counts) if cnt > 0
    }

    # --- Datasets ---
    train_dataset = BeeDataset(
        train_paths, train_labels,
        augment_factors=augment_factors,
        apply_augmentation=True,
        max_per_class=max_per_class,
        image_cache=True,
    )
    val_dataset = BeeDataset(
        val_paths, val_labels,
        apply_augmentation=False,
        max_per_class=1000,
        image_cache=True,
    )

    # --- Weighted sampler ---
    train_labels_ext = [train_labels[i] for i in train_dataset.indices]
    class_weights    = 1.0 / (np.bincount(train_labels_ext, minlength=num_classes) + 1e-6)
    sample_weights   = [class_weights[l] for l in train_labels_ext]
    sampler          = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- Training loop ---
    best_f1    = 0.0
    patience   = 5
    no_improve = 0

    for epoch in range(num_epochs):

        # Training
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                val_preds.append(model(imgs.to(DEVICE)).cpu())
                val_lbls.append(lbls)

        val_f1 = compute_f1(torch.cat(val_preds), torch.cat(val_lbls))

        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            try:
                if val_f1 > study.best_value:
                    torch.save(model.state_dict(), "best_model_optuna.pth")
                    print(f"  New global best -- trial {trial.number} "
                          f"epoch {epoch + 1}  F1={val_f1:.4f}")
            except ValueError:
                torch.save(model.state_dict(), "best_model_optuna.pth")
        else:
            no_improve += 1

        # Pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            _cleanup(model, train_dataset, val_dataset, train_loader, val_loader)
            raise optuna.exceptions.TrialPruned()

        # Early stopping
        if no_improve >= patience:
            break

    _cleanup(model, train_dataset, val_dataset, train_loader, val_loader)
    return best_f1


def _cleanup(*objects):
    """Release GPU/MPS memory between trials."""
    for obj in objects:
        del obj
    torch.mps.empty_cache()
    gc.collect()

# ==============================================================================
# CALLBACK
# ==============================================================================

def print_trial_callback(study: optuna.Study, trial: optuna.Trial):
    pruned = trial.state == optuna.trial.TrialState.PRUNED
    result = "PRUNED" if pruned else f"F1={trial.value:.4f}"
    print(f"\n{'='*55}")
    print(f"Trial {trial.number:3d} | {result}")
    for k, v in trial.params.items():
        fmt = f"{v:.2e}" if isinstance(v, float) else str(v)
        print(f"  {k:<20s} : {fmt}")
    print(f"  -> Global best : F1={study.best_value:.4f} (trial {study.best_trial.number})")
    print(f"{'='*55}")

# ==============================================================================
# MAIN
# ==============================================================================

gc.collect()
torch.mps.empty_cache()

study = optuna.create_study(
    direction      = "maximize",
    sampler        = optuna.samplers.TPESampler(seed=SEED),
    pruner         = optuna.pruners.MedianPruner(n_warmup_steps=3),
    storage        = "sqlite:///optuna_study.db",
    study_name     = "bee_classification",
    load_if_exists = True,   # resume if study already exists
)

study.optimize(
    objective,
    n_trials          = 50,
    show_progress_bar = True,
    callbacks         = [print_trial_callback],
)

print("\n===== Best hyperparameters =====")
print(f"  Val F1 : {study.best_value:.4f}")
for k, v in study.best_params.items():
    print(f"  {k:<20s} : {v}")