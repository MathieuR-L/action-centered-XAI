import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, KernelShap, LayerGradCam, Lime
from medmnist import DermaMNIST
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


FEATURE_NAMES = [
    "image",
    "biomarker_score",
    "age_risk",
    "hospital_bias",
    "admin_noise",
]
NUM_TAB_FEATURES = 4
NUM_CLASSES = 7
IMAGE_SIZE = 28


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("Arrays must share the same shape.")
    if len(x) < 2:
        return float("nan")
    xranks = np.argsort(np.argsort(x)).astype(np.float64)
    yranks = np.argsort(np.argsort(y)).astype(np.float64)
    xranks -= xranks.mean()
    yranks -= yranks.mean()
    denom = np.linalg.norm(xranks) * np.linalg.norm(yranks)
    if denom == 0:
        return float("nan")
    return float(np.dot(xranks, yranks) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def stable_seed(split: str, index: int, ood_hospital_bias: bool) -> int:
    base = 13 if split == "train" else 29 if split == "val" else 47
    modifier = 97 if ood_hospital_bias else 0
    return base * 100_003 + index * 17 + modifier


class ControlledDermaMNIST(Dataset):
    def __init__(
        self,
        split: str,
        *,
        transform: transforms.Compose,
        ood_hospital_bias: bool = False,
        limit: int | None = None,
    ) -> None:
        self.base = DermaMNIST(split=split, download=True, size=IMAGE_SIZE)
        self.transform = transform
        self.split = split
        self.ood_hospital_bias = ood_hospital_bias
        self.indices = list(range(len(self.base)))
        if limit is not None:
            self.indices = self.indices[:limit]
        self.biomarker_centers = np.array([0.15, 0.25, 0.35, 0.50, 0.62, 0.78, 0.90], dtype=np.float32)
        self.age_centers = np.array([0.72, 0.18, 0.61, 0.28, 0.83, 0.41, 0.55], dtype=np.float32)
        self.hospital_centers = np.array([0.10, 0.24, 0.38, 0.52, 0.66, 0.80, 0.94], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def _tabular_features(self, label: int, index: int) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.split, index, self.ood_hospital_bias))
        biomarker = np.clip(self.biomarker_centers[label] + rng.normal(0.0, 0.12), 0.0, 1.0)
        age_risk = np.clip(self.age_centers[label] + rng.normal(0.0, 0.16), 0.0, 1.0)
        if self.ood_hospital_bias:
            hospital_bias = rng.uniform(0.0, 1.0)
        else:
            hospital_bias = np.clip(self.hospital_centers[label] + rng.normal(0.0, 0.035), 0.0, 1.0)
        admin_noise = rng.uniform(0.0, 1.0)
        return np.asarray([biomarker, age_risk, hospital_bias, admin_noise], dtype=np.float32)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_index = self.indices[item]
        image, label = self.base[base_index]
        image_tensor = self.transform(image)
        label_value = int(label[0])
        tabular = torch.tensor(self._tabular_features(label_value, base_index), dtype=torch.float32)
        return image_tensor, tabular, torch.tensor(label_value, dtype=torch.long)


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.proj = nn.Linear(64, hidden_dim)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(image))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        feature_map = F.relu(self.conv3(x))
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return self.proj(pooled), feature_map


class MultimodalAttentionNet(nn.Module):
    def __init__(self, hidden_dim: int = 96, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim)
        self.tabular_projections = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(NUM_TAB_FEATURES)])
        self.feature_embeddings = nn.Parameter(torch.randn(NUM_TAB_FEATURES, hidden_dim) * 0.02)
        self.image_token_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, NUM_CLASSES)

    def _tabular_tokens(self, tabular: torch.Tensor) -> torch.Tensor:
        tokens = []
        for index, projection in enumerate(self.tabular_projections):
            value = tabular[:, index : index + 1]
            token = projection(value) + self.feature_embeddings[index]
            tokens.append(token)
        return torch.stack(tokens, dim=1)

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_token, feature_map = self.image_encoder(image)
        image_token = image_token.unsqueeze(1) + self.image_token_embedding
        tabular_tokens = self._tabular_tokens(tabular)
        cls = self.cls_token.expand(image.shape[0], -1, -1)
        tokens = torch.cat([cls, image_token, tabular_tokens], dim=1)

        normed = self.ln1(tokens)
        attn_out, attn_weights = self.attn(
            normed,
            normed,
            normed,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ln2(tokens))
        cls_state = tokens[:, 0]
        logits = self.head(cls_state)

        if return_details:
            details = {
                "attn_weights": attn_weights,
                "feature_map": feature_map,
                "joint_embedding": cls_state,
            }
            return logits, details
        return logits

    def joint_embedding(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        _, details = self.forward(image, tabular, return_details=True)
        return details["joint_embedding"]


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    causal_lambda: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    *,
    seed: int,
    causal_invariance: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_acc = -math.inf
    history: Dict[str, float] = {}

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, tabular, labels in train_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, tabular)
            loss = F.cross_entropy(logits, labels)

            if causal_invariance:
                shuffled = tabular.clone()
                perm = torch.randperm(shuffled.shape[0], device=device)
                shuffled[:, 2] = shuffled[perm, 2]
                logits_cf = model(images, shuffled)
                consistency = F.mse_loss(
                    F.softmax(logits, dim=1),
                    F.softmax(logits_cf, dim=1),
                )
                loss = loss + config.causal_lambda * consistency

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_acc = running_correct / max(running_total, 1)
        val_metrics = evaluate_model(model, val_loader, device)
        history = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(running_total, 1),
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for images, tabular, labels in loader:
            logits = model(images.to(device), tabular.to(device))
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    labels_arr = np.asarray(all_labels)
    preds_arr = np.asarray(all_preds)
    accuracy = float((labels_arr == preds_arr).mean())

    macro_f1_values = []
    for class_index in range(NUM_CLASSES):
        tp = np.sum((preds_arr == class_index) & (labels_arr == class_index))
        fp = np.sum((preds_arr == class_index) & (labels_arr != class_index))
        fn = np.sum((preds_arr != class_index) & (labels_arr == class_index))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            macro_f1_values.append(0.0)
        else:
            macro_f1_values.append(2 * precision * recall / (precision + recall))

    return {"accuracy": accuracy, "macro_f1": float(np.mean(macro_f1_values))}


def build_feature_masks(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    patch = 7
    mask = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.long)
    feature_id = 0
    for row in range(0, IMAGE_SIZE, patch):
        for col in range(0, IMAGE_SIZE, patch):
            mask[:, :, row : row + patch, col : col + patch] = feature_id
            feature_id += 1
    tabular_mask = torch.arange(feature_id, feature_id + NUM_TAB_FEATURES, dtype=torch.long).view(1, NUM_TAB_FEATURES)
    return mask.to(device), tabular_mask.to(device)


def summarize_attributions(image_attr: torch.Tensor, tabular_attr: torch.Tensor) -> np.ndarray:
    image_score = image_attr.abs().mean().item()
    feature_scores = tabular_attr.abs().flatten().detach().cpu().numpy()
    return np.asarray([image_score, *feature_scores.tolist()], dtype=np.float64)


def normalized_heatmap(heatmap: torch.Tensor) -> np.ndarray:
    array = heatmap.detach().cpu().float().numpy().reshape(-1)
    if np.allclose(array.max(), array.min()):
        return np.zeros_like(array, dtype=np.float64)
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)
    return array.astype(np.float64)


def manual_ablation_scores(
    model: nn.Module,
    image: torch.Tensor,
    tabular: torch.Tensor,
    target: int,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        base_logit = model(image, tabular)[0, target].item()
        blank_image = torch.zeros_like(image)
        masked_image_logit = model(blank_image, tabular)[0, target].item()
        scores = [max(0.0, base_logit - masked_image_logit)]
        for feature_index in range(NUM_TAB_FEATURES):
            masked_tabular = tabular.clone()
            masked_tabular[:, feature_index] = 0.0
            masked_logit = model(image, masked_tabular)[0, target].item()
            scores.append(max(0.0, base_logit - masked_logit))
    return np.asarray(scores, dtype=np.float64)


def accuracy_after_feature_mask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_index: int,
) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, tabular, labels in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            if feature_index == 0:
                images = torch.zeros_like(images)
            else:
                tabular[:, feature_index - 1] = 0.0
            logits = model(images, tabular)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def collect_joint_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, tabular, batch_labels in loader:
            joint = model.joint_embedding(images.to(device), tabular.to(device)).detach().cpu().numpy()
            embeddings.append(joint)
            labels.append(batch_labels.numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def retrieval_metrics(
    model: nn.Module,
    train_loader: DataLoader,
    sample_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    train_embeddings, train_labels = collect_joint_embeddings(model, train_loader, device)
    nn_index = NearestNeighbors(metric="cosine", n_neighbors=5)
    nn_index.fit(train_embeddings)

    latencies = []
    purities = []
    model.eval()
    with torch.no_grad():
        for images, tabular, labels in sample_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.numpy()
            for sample_index in range(images.shape[0]):
                query_image = images[sample_index : sample_index + 1]
                query_tab = tabular[sample_index : sample_index + 1]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                query_embedding = model.joint_embedding(query_image, query_tab).detach().cpu().numpy()
                _, neighbor_ids = nn_index.kneighbors(query_embedding)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000.0)
                neighbor_labels = train_labels[neighbor_ids[0]]
                purities.append(float(np.mean(neighbor_labels == labels[sample_index])))
    return {
        "latency_ms_mean": float(np.mean(latencies)),
        "evidence_purity": float(np.mean(purities)),
    }


def compute_explanations(
    model: nn.Module,
    model_variant: str,
    paired_model: nn.Module,
    sample_loader: DataLoader,
    ood_loader: DataLoader,
    train_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    paired_model.eval()
    image_mask, tabular_mask = build_feature_masks(device)
    ig = IntegratedGradients(model)
    lime = Lime(model)
    kshap = KernelShap(model)
    gradcam = LayerGradCam(model, model.image_encoder.conv3)

    explanation_vectors: Dict[str, List[np.ndarray]] = {
        "attention": [],
        "integrated_gradients": [],
        "lime": [],
        "kernel_shap": [],
        "modality_ablation": [],
    }
    latencies: Dict[str, List[float]] = {key: [] for key in explanation_vectors}
    latencies["gradcam"] = []
    heatmap_stability: List[float] = []
    heatmap_leakage: List[float] = []
    vector_stability: Dict[str, List[float]] = {key: [] for key in explanation_vectors}
    ig_steps = 32
    perturbation_samples = 128

    baseline_image = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
    baseline_tabular = torch.zeros((1, NUM_TAB_FEATURES), device=device)

    for images, tabular, labels in sample_loader:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)
        for sample_index in range(images.shape[0]):
            image = images[sample_index : sample_index + 1]
            tab = tabular[sample_index : sample_index + 1]
            target = int(labels[sample_index].item())

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _, details = model(image, tab, return_details=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies["attention"].append((time.perf_counter() - start) * 1000.0)
            attention = details["attn_weights"].mean(dim=1)[0, 0, 1:].detach().cpu().numpy()
            explanation_vectors["attention"].append(attention.astype(np.float64))

            _, paired_details = paired_model(image, tab, return_details=True)
            paired_attention = paired_details["attn_weights"].mean(dim=1)[0, 0, 1:].detach().cpu().numpy()
            vector_stability["attention"].append(cosine_similarity(attention, paired_attention))

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            ig_image, ig_tab = ig.attribute(
                (image, tab),
                baselines=(baseline_image, baseline_tabular),
                target=target,
                n_steps=ig_steps,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies["integrated_gradients"].append((time.perf_counter() - start) * 1000.0)
            ig_vector = summarize_attributions(ig_image, ig_tab)
            explanation_vectors["integrated_gradients"].append(ig_vector)

            paired_ig = IntegratedGradients(paired_model)
            paired_ig_image, paired_ig_tab = paired_ig.attribute(
                (image, tab),
                baselines=(baseline_image, baseline_tabular),
                target=target,
                n_steps=ig_steps,
            )
            paired_ig_vector = summarize_attributions(paired_ig_image, paired_ig_tab)
            vector_stability["integrated_gradients"].append(cosine_similarity(ig_vector, paired_ig_vector))

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            lime_image, lime_tab = lime.attribute(
                (image, tab),
                baselines=(baseline_image, baseline_tabular),
                target=target,
                feature_mask=(image_mask, tabular_mask),
                n_samples=perturbation_samples,
                perturbations_per_eval=16,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies["lime"].append((time.perf_counter() - start) * 1000.0)
            lime_vector = summarize_attributions(lime_image, lime_tab)
            explanation_vectors["lime"].append(lime_vector)

            paired_lime = Lime(paired_model)
            paired_lime_image, paired_lime_tab = paired_lime.attribute(
                (image, tab),
                baselines=(baseline_image, baseline_tabular),
                target=target,
                feature_mask=(image_mask, tabular_mask),
                n_samples=perturbation_samples,
                perturbations_per_eval=16,
            )
            paired_lime_vector = summarize_attributions(paired_lime_image, paired_lime_tab)
            vector_stability["lime"].append(cosine_similarity(lime_vector, paired_lime_vector))

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            shap_image, shap_tab = kshap.attribute(
                (image, tab),
                baselines=(baseline_image, baseline_tabular),
                target=target,
                feature_mask=(image_mask, tabular_mask),
                n_samples=perturbation_samples,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies["kernel_shap"].append((time.perf_counter() - start) * 1000.0)
            shap_vector = summarize_attributions(shap_image, shap_tab)
            explanation_vectors["kernel_shap"].append(shap_vector)

            paired_shap = KernelShap(paired_model)
            paired_shap_image, paired_shap_tab = paired_shap.attribute(
                (image, tab),
                baselines=(baseline_image, baseline_tabular),
                target=target,
                feature_mask=(image_mask, tabular_mask),
                n_samples=perturbation_samples,
            )
            paired_shap_vector = summarize_attributions(paired_shap_image, paired_shap_tab)
            vector_stability["kernel_shap"].append(cosine_similarity(shap_vector, paired_shap_vector))

            start = time.perf_counter()
            ablation_vector = manual_ablation_scores(model, image, tab, target)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies["modality_ablation"].append((time.perf_counter() - start) * 1000.0)
            explanation_vectors["modality_ablation"].append(ablation_vector)
            paired_ablation = manual_ablation_scores(paired_model, image, tab, target)
            vector_stability["modality_ablation"].append(cosine_similarity(ablation_vector, paired_ablation))

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            heatmap = gradcam.attribute(image, target=target, additional_forward_args=tab)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies["gradcam"].append((time.perf_counter() - start) * 1000.0)
            heatmap_vector = normalized_heatmap(heatmap)

            paired_gradcam = LayerGradCam(paired_model, paired_model.image_encoder.conv3)
            paired_heatmap = paired_gradcam.attribute(image, target=target, additional_forward_args=tab)
            paired_heatmap_vector = normalized_heatmap(paired_heatmap)
            heatmap_stability.append(cosine_similarity(heatmap_vector, paired_heatmap_vector))

            swapped_tab = tab.clone()
            swapped_tab[:, 2] = 1.0 - swapped_tab[:, 2]
            leaked_heatmap = gradcam.attribute(image, target=target, additional_forward_args=swapped_tab)
            leaked_heatmap_vector = normalized_heatmap(leaked_heatmap)
            heatmap_leakage.append(1.0 - cosine_similarity(heatmap_vector, leaked_heatmap_vector))

    base_ood_acc = evaluate_model(model, ood_loader, device)["accuracy"]
    feature_importance = []
    for feature_index in range(len(FEATURE_NAMES)):
        masked_acc = accuracy_after_feature_mask(model, ood_loader, device, feature_index)
        feature_importance.append(base_ood_acc - masked_acc)
    feature_importance_arr = np.asarray(feature_importance, dtype=np.float64)

    causal_alignment = {}
    mean_vectors = {}
    for method_name, vectors in explanation_vectors.items():
        averaged = np.mean(np.stack(vectors, axis=0), axis=0)
        mean_vectors[method_name] = {
            name: float(value) for name, value in zip(FEATURE_NAMES, averaged.tolist())
        }
        causal_alignment[method_name] = spearman_corr(averaged, feature_importance_arr)

    retrieval = retrieval_metrics(model, train_loader, sample_loader, device)

    return {
        "variant": {"name": model_variant},
        "feature_importance_ood": {
            name: float(value) for name, value in zip(FEATURE_NAMES, feature_importance_arr.tolist())
        },
        "retrieval": retrieval,
        "methods": {
            "attention": {
                "latency_ms_mean": float(np.mean(latencies["attention"])),
                "stability_cosine_mean": float(np.mean(vector_stability["attention"])),
                "causal_alignment_spearman": float(causal_alignment["attention"]),
                "mean_scores": mean_vectors["attention"],
            },
            "integrated_gradients": {
                "latency_ms_mean": float(np.mean(latencies["integrated_gradients"])),
                "stability_cosine_mean": float(np.mean(vector_stability["integrated_gradients"])),
                "causal_alignment_spearman": float(causal_alignment["integrated_gradients"]),
                "mean_scores": mean_vectors["integrated_gradients"],
            },
            "lime": {
                "latency_ms_mean": float(np.mean(latencies["lime"])),
                "stability_cosine_mean": float(np.mean(vector_stability["lime"])),
                "causal_alignment_spearman": float(causal_alignment["lime"]),
                "mean_scores": mean_vectors["lime"],
            },
            "kernel_shap": {
                "latency_ms_mean": float(np.mean(latencies["kernel_shap"])),
                "stability_cosine_mean": float(np.mean(vector_stability["kernel_shap"])),
                "causal_alignment_spearman": float(causal_alignment["kernel_shap"]),
                "mean_scores": mean_vectors["kernel_shap"],
            },
            "modality_ablation": {
                "latency_ms_mean": float(np.mean(latencies["modality_ablation"])),
                "stability_cosine_mean": float(np.mean(vector_stability["modality_ablation"])),
                "causal_alignment_spearman": float(causal_alignment["modality_ablation"]),
                "mean_scores": mean_vectors["modality_ablation"],
            },
            "gradcam": {
                "latency_ms_mean": float(np.mean(latencies["gradcam"])),
                "stability_cosine_mean": float(np.mean(heatmap_stability)),
                "leakage_shift_mean": float(np.mean(heatmap_leakage)),
            },
        },
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def subset_for_explanations(dataset: Dataset, size: int) -> Dataset:
    return Subset(dataset, list(range(min(size, len(dataset)))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled experiments for ACE-aligned multimodal XAI assessment.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--explain-samples", type=int, default=12)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="experiments/results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ControlledDermaMNIST(
        "train",
        transform=transform,
        ood_hospital_bias=False,
        limit=args.train_limit,
    )
    val_dataset = ControlledDermaMNIST("val", transform=transform, ood_hospital_bias=False)
    test_dataset = ControlledDermaMNIST("test", transform=transform, ood_hospital_bias=False)
    ood_test_dataset = ControlledDermaMNIST("test", transform=transform, ood_hospital_bias=True)

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
    eval_train_loader = make_loader(train_dataset, args.batch_size, shuffle=False)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False)
    ood_loader = make_loader(ood_test_dataset, args.batch_size, shuffle=False)
    explanation_loader = make_loader(subset_for_explanations(ood_test_dataset, args.explain_samples), 1, shuffle=False)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        causal_lambda=0.75,
    )

    baseline_model = MultimodalAttentionNet().to(device)
    baseline_model, baseline_history = train_model(
        baseline_model,
        train_loader,
        val_loader,
        device,
        config,
        seed=args.seed,
        causal_invariance=False,
    )

    paired_baseline_model = MultimodalAttentionNet().to(device)
    paired_baseline_model, _ = train_model(
        paired_baseline_model,
        train_loader,
        val_loader,
        device,
        config,
        seed=args.seed + 1,
        causal_invariance=False,
    )

    causal_model = MultimodalAttentionNet().to(device)
    causal_model, causal_history = train_model(
        causal_model,
        train_loader,
        val_loader,
        device,
        config,
        seed=args.seed,
        causal_invariance=True,
    )

    baseline_metrics = {
        "id_test": evaluate_model(baseline_model, test_loader, device),
        "ood_test": evaluate_model(baseline_model, ood_loader, device),
        "history": baseline_history,
    }
    causal_metrics = {
        "id_test": evaluate_model(causal_model, test_loader, device),
        "ood_test": evaluate_model(causal_model, ood_loader, device),
        "history": causal_history,
    }

    baseline_explanations = compute_explanations(
        baseline_model,
        "baseline",
        paired_baseline_model,
        explanation_loader,
        ood_loader,
        eval_train_loader,
        device,
    )
    causal_explanations = compute_explanations(
        causal_model,
        "causal_invariance",
        baseline_model,
        explanation_loader,
        ood_loader,
        eval_train_loader,
        device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "ace_experiment_results.json"
    payload = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
            "explain_samples": args.explain_samples,
            "train_limit": args.train_limit,
            "device": str(device),
        },
        "dataset": {
            "base_dataset": "DermaMNIST",
            "task": "medical image classification with controlled multimodal tabular covariates",
            "feature_names": FEATURE_NAMES,
        },
        "models": {
            "baseline": baseline_metrics,
            "causal_invariance": causal_metrics,
        },
        "experiments": {
            "baseline": baseline_explanations,
            "causal_invariance": causal_explanations,
        },
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
