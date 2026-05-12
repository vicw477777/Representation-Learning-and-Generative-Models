from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import zipfile
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm


BASE_DATA_PATH = Path(".") / "data"
ZIP_PATH = BASE_DATA_PATH / "part1.zip"
DATA_PATH = BASE_DATA_PATH / "part1"


FRUIT_CLASS_NAMES = [
    "granny_smith",
    "strawberry",
    "orange",
    "lemon",
    "fig",
    "pineapple",
    "banana",
    "jackfruit",
    "custard_apple",
    "pomegranate",
]

datas = {}


def download_data():
    BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        file_id = "1qNFjQIBck90I41aiJMpZR70NZRHQtEqE"
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            output=str(ZIP_PATH),
            quiet=True,
        )

    if not DATA_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(str(BASE_DATA_PATH))


def load_data(split_name: str):
    download_data()
    images = np.load(DATA_PATH / f"{split_name}_images.npy")
    labels = np.load(DATA_PATH / f"{split_name}_labels.npy")
    return images, labels


def get_data(split_name: str):
    if split_name not in datas:
        datas[split_name] = load_data(split_name)
    return datas[split_name]


class FruitDataset(torch.utils.data.Dataset):
    def __init__(self, split_name: str, transform: Optional[Callable] = None):
        self.images, self.labels = get_data(split_name)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return int(np.max(self.labels) + 1)


def visualize_samples(
    split_name: str,
    num_rows: int = 2,
    figsize: tuple[int, int] = (10, 5),
    seed: int = 0,
):
    dataset = FruitDataset(split_name)
    rng = np.random.RandomState(seed)
    _, axes = plt.subplots(num_rows, dataset.num_classes // num_rows, figsize=figsize)
    for class_idx in range(dataset.num_classes):
        row = class_idx // (dataset.num_classes // num_rows)
        col = class_idx % (dataset.num_classes // num_rows)
        class_indices = np.where(dataset.labels == class_idx)[0]
        random_idx = int(rng.choice(class_indices))
        img, _ = dataset[random_idx]
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{FRUIT_CLASS_NAMES[class_idx]}")
        axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(f"part1_{split_name}_samples.png")
    plt.close()


def create_feature_extractor(model_name: str, pretrained_cfg: str, device: str = "cuda:0"):
    model = timm.create_model(model_name, pretrained=True, pretrained_cfg=pretrained_cfg)
    model.transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.head = nn.Identity()
    return model.eval().to(device)


def get_features(
    split_name: str,
    feature_extractor: torch.nn.Module,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    feature_extractor.eval()
    features = None

    ### YOUR CODE STARTS HERE ###
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    feats = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"Extracting {split_name} features"):
            images = images.to(device, non_blocking=True)
            out = feature_extractor(images)           # (B, D)
            out = out.detach().float().cpu().numpy()  # move to CPU numpy
            feats.append(out)

    features = np.concatenate(feats, axis=0)          # (N, D)
    ### YOUR CODE ENDS HERE ###

    return features, dataset.labels, dataset.num_classes


class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, num_classes: int):
        assert features.ndim == 2, f"Expected (N, D), got {features.shape}"
        self.features = features.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.num_classes = int(num_classes)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.features[index]).float()
        y = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.labels)

    @classmethod
    def create(cls, split_name: str, feature_extractor: torch.nn.Module, **kwargs):
        features, labels, num_classes = get_features(split_name, feature_extractor, **kwargs)
        return cls(features, labels, num_classes)


def visualize_features_tsne(
    features_dataset,
    title: str = "Features t-SNE",
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 0,
):
    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    features_2d = tsne.fit_transform(features_dataset.features)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=features_dataset.labels,
        cmap="tab10",
        alpha=0.7,
    )
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=FRUIT_CLASS_NAMES,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(f"part1_{title.replace(' ', '_').lower()}.png")
    plt.close()


def train_linear_probe(
    features_dataset: FeaturesDataset,
    num_epochs: int = 32,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    linear_probe = None
    epoch_losses = []

    ### YOUR CODE STARTS HERE ###
    D = int(features_dataset.features.shape[1])
    C = int(features_dataset.num_classes)

    # A single linear layer: feature_dim -> num_classes
    linear_probe = nn.Linear(D, C).to(device)

    dataloader = torch.utils.data.DataLoader(
        features_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        linear_probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in range(num_epochs):
        linear_probe.train()
        running_loss = 0.0
        n = 0

        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = linear_probe(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(x.shape[0])
            running_loss += float(loss.item()) * bs
            n += bs

        epoch_losses.append(running_loss / max(n, 1))
    ### YOUR CODE ENDS HERE ###

    return linear_probe, epoch_losses


def train_finetune_probe(
    split_name: str,
    feature_extractor: torch.nn.Module,
    pretrained_linear_probe: torch.nn.Module = None,
    num_epochs: int = 20,
    batch_size: int = 32,
    feature_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = "cuda:0",
):
    ### YOUR CODE STARTS HERE ###
    dataset = FruitDataset(split_name, transform=feature_extractor.transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )

    feature_extractor = feature_extractor.to(device)

    # Infer feature dimension D with a single forward pass
    feature_extractor.eval()
    with torch.no_grad():
        x0, _ = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        D = int(feature_extractor(x0).shape[1])
    C = int(dataset.num_classes)

    # Build classification head and optionally initialise from the linear probe
    head = nn.Linear(D, C).to(device)
    if pretrained_linear_probe is not None:
        head.load_state_dict(pretrained_linear_probe.state_dict())

    # Wrap backbone + head into a single model
    class FinetuneModel(nn.Module):
        def __init__(self, extractor: nn.Module, head: nn.Module):
            super().__init__()
            self.extractor = extractor
            self.head = head

        def forward(self, x):
            f = self.extractor(x)
            return self.head(f)

    model = FinetuneModel(feature_extractor, head).to(device)

    criterion = nn.CrossEntropyLoss()
    # Use a differential learning rate: small LR for backbone, larger for head
    optimizer = torch.optim.AdamW(
        [
            {"params": model.extractor.parameters(), "lr": feature_lr},
            {"params": model.head.parameters(),      "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    epoch_losses = []
    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        n = 0

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(images.shape[0])
            running_loss += float(loss.item()) * bs
            n += bs

        epoch_losses.append(running_loss / max(n, 1))
    ### YOUR CODE ENDS HERE ###

    return model, epoch_losses


def plot_losses(losses, name):
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training loss for {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"part1_{name}_training_loss.png")
    plt.show()
    plt.close()


def evaluate_linear(probe, features_dataset, device="cuda:0"):
    probe.eval()
    loader = torch.utils.data.DataLoader(
        features_dataset, batch_size=1024, shuffle=False, num_workers=4,
        pin_memory=device.startswith("cuda"),
    )
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = probe(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total   += int(y.numel())
    return correct / max(total, 1)


def evaluate_finetune(model, split_name, fx, device="cuda:0"):
    model.eval()
    ds = FruitDataset(split_name, transform=fx.transform)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=4,
        pin_memory=device.startswith("cuda"),
    )
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total   += int(labels.numel())
    return correct / max(total, 1)
