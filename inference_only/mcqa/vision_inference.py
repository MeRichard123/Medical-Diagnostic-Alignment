from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm

from utils import train_test_split


"""
vit_b16_head_only
alexnet
resnet101
swin_v2_b
vgg19
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed_iuxray_mcqa_dataset.csv"
IMAGE_COL = "image_1"
LABEL_COL = "copt"
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PRETRAINED = True
TRAIN_HEAD_ONLY = True
MODEL_NAME = "alexnet"
RESULTS_PATH = PROJECT_ROOT / "results" / f"{MODEL_NAME}_inference_results.json"


class VisionDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_path, label_idx = self.records[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label_idx


def resolve_image_path(image_name):
    base = PROJECT_ROOT
    candidates = [
        base / "data" / "images" / "processed" / image_name,
        base / "images" / "processed" / image_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None



def get_vision_model(pretrained, num_classes, train_head_only=True):
    match MODEL_NAME:
        case "vit_b16_head_only":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vit_b_16(weights=weights)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        case "alexnet":
            weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.alexnet(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        case "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet101(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        case "swin_v2_b":
            weights = models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.swin_v2_b(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        case "vgg19":
            weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vgg19(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        case _:
            raise ValueError(f"Unsupported model name: {MODEL_NAME}")

    if train_head_only:
        for param in model.parameters():
            param.requires_grad = False

        for param in list(model.children())[-2:]:
            for param in param.parameters():
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    return model


def build_records(df, label_to_idx):
    records = []
    skipped = 0
    for _, row in df.iterrows():
        image_name = str(row[IMAGE_COL]).strip()
        label_name = str(row[LABEL_COL]).strip()
        image_path = resolve_image_path(image_name)
        if image_path is None:
            skipped += 1
            continue
        records.append((image_path, label_to_idx[label_name]))
    if skipped:
        print(f"Skipped {skipped} rows with missing image files.")
    return records


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def predict_with_probs(model, image_path, transform, idx_to_label, device):
    model.eval()
    with torch.no_grad():
        x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_label[pred_idx]
    prob_dict = {idx_to_label[i]: float(probs[i].item()) for i in range(len(idx_to_label))}
    return pred_label, prob_dict


if __name__ == "__main__":
    torch.manual_seed(RANDOM_STATE)

    data = pd.read_csv(CSV_PATH)
    data = data[['uid',IMAGE_COL, LABEL_COL]].dropna()
    data[IMAGE_COL] = data[IMAGE_COL].astype(str).str.strip()
    data[LABEL_COL] = data[LABEL_COL].astype(str).str.strip()
    data['uid'] = data['uid'].astype(str).str.strip()

    train_df, test_df = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    labels = sorted(data[LABEL_COL].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    if PRETRAINED:
        match MODEL_NAME:
            case "vit_b16_head_only":
                transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            case "alexnet":
                transform = models.AlexNet_Weights.IMAGENET1K_V1.transforms()
            case "resnet101":
                transform = models.ResNet101_Weights.IMAGENET1K_V1.transforms()
            case "swin_v2_b":  
                transform = models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms()
            case "vgg19":
                transform = models.VGG19_Weights.IMAGENET1K_V1.transforms()
            case _:
                raise ValueError(f"Unsupported model name: {MODEL_NAME}")
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    train_records = build_records(train_df, label_to_idx)
    test_records = build_records(test_df, label_to_idx)

    if not train_records:
        raise ValueError(
            "No training records found after path resolution. "
            "Verify image paths under data/images/processed and CSV image names."
        )
    if not test_records:
        raise ValueError(
            "No test records found after path resolution. "
            "Verify image paths under data/images/processed and CSV image names."
        )

    train_loader = DataLoader(VisionDataset(train_records, transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(VisionDataset(test_records, transform), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vision_model(PRETRAINED, len(labels), TRAIN_HEAD_ONLY).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Train: {len(train_records)} | Test: {len(test_records)} | Classes: {len(labels)}")
    print(f"Device: {device}")

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    if test_records:
        sample_image_path, sample_label_idx = test_records[0]
        model.eval()
        with torch.no_grad():
            x = transform(Image.open(sample_image_path).convert("RGB")).unsqueeze(0).to(device)
            pred_idx = model(x).argmax(dim=1).item()
        print(f"Sample image: {sample_image_path.name}")
        print(f"Ground truth: {idx_to_label[sample_label_idx]}")
        print(f"Predicted: {idx_to_label[pred_idx]}")

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        image_name = str(row[IMAGE_COL]).strip()
        image_path = resolve_image_path(image_name)
        if image_path is None:
            continue

        pred_label, prob_dict = predict_with_probs(model, image_path, transform, idx_to_label, device)
        result = {
            "id": row["uid"],
            "predicted_diagnosis": pred_label,
            "ground_truth": str(row[LABEL_COL]).strip(),
            "probabilities": prob_dict,
        }
        results.append(result)

    output_path = Path(RESULTS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved test predictions to: {output_path}")
