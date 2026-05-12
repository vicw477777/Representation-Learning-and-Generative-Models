import numpy as np
import matplotlib.pyplot as plt

from representations import (
    create_feature_extractor,
    visualize_samples,
    FeaturesDataset,
    visualize_features_tsne,
    train_linear_probe,
    train_finetune_probe,
    evaluate_linear,
    evaluate_finetune,
    plot_losses,
)

device = "cuda:0"

feature_extractors = {
    "clip": create_feature_extractor("vit_base_patch16_clip_224", "openai", device=device),
    "dino": create_feature_extractor("vit_base_patch16_224", "dino", device=device),
    "mae":  create_feature_extractor("vit_base_patch16_224", "mae", device=device),
}

# Visualise some samples
visualize_samples("photo_val", seed=0)

# Extract features for training set
train_features_datasets = {}
for method_name, fx in feature_extractors.items():
    print(f"Extracting features for training set using {method_name}...")
    train_features_datasets[method_name] = FeaturesDataset.create(
        "photo_train", fx, device=device
    )
    assert train_features_datasets[method_name].features.shape[0] == 13000
    assert train_features_datasets[method_name].features.ndim == 2

# Visualize features using t-SNE
for method_name in feature_extractors.keys():
    print(f"Visualizing features for {method_name}...")
    visualize_features_tsne(
        train_features_datasets[method_name],
        title=f"Features t-SNE ({method_name})"
    )

# Linear probe training
linear_probes = {}
for method_name in feature_extractors.keys():
    print(f"Training linear probe for {method_name}...")
    linear_probes[method_name], losses = train_linear_probe(
        train_features_datasets[method_name], device=device
    )
    plot_losses(losses, method_name)

# Evaluate linear probes on photo_val (implement evaluate_linear from scratch)
for method_name, probe in linear_probes.items():
    val_feats = FeaturesDataset.create("photo_val", feature_extractors[method_name], device=device)
    acc = evaluate_linear(probe, val_feats, device=device)
    print(f"{method_name} linear probe photo-val accuracy: {acc:.4f}")

# Finetune probe training (init from linear probe)
finetuned_models = {}
for method_name, fx in feature_extractors.items():
    print(f"Fine-tuning {method_name} (init from linear probe)...")
    finetuned_models[method_name], ft_losses = train_finetune_probe(
        "photo_train",
        fx,
        pretrained_linear_probe=linear_probes[method_name],
        device=device,
    )
    plot_losses(ft_losses, f"{method_name}_finetune")

    # Extract features from the fine-tuned backbone and visualise with t-SNE
    print(f"Extracting fine-tuned features for t-SNE ({method_name})...")
    ft_feats = FeaturesDataset.create("photo_train", fx, device=device)
    visualize_features_tsne(ft_feats, title=f"Fine-tuned Features t-SNE ({method_name})")

# Evaluate finetuned models on photo_val (implement evaluate_finetune from scratch)
for method_name, fx in feature_extractors.items():
    print(f"Evaluating {method_name} finetuned model...")
    acc = evaluate_finetune(finetuned_models[method_name], "photo_val", fx, device=device)
    print(f"{method_name} finetune photo-val accuracy: {acc:.4f}")
