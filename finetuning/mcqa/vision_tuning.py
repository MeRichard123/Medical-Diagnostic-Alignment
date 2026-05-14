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
UNFREEZE_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
PRETRAINED = True
TRAIN_HEAD_ONLY = True
ENABLE_TWO_PHASE_FINETUNE = True
HEAD_EPOCHS = 3
UNFREEZE_EPOCHS = 7
MODEL_NAME = "alexnet"
RESULTS_PATH = PROJECT_ROOT / "results" / f"{MODEL_NAME}-finetuned_inference_results.json"
MODEL_SAVE_PATH = PROJECT_ROOT / "finetuning" / "tuned" / f"{MODEL_NAME}_finetuned.pth"


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



def _set_classifier_trainable(model, trainable):
    if MODEL_NAME == "vit_b16_head_only":
        for param in model.heads.head.parameters():
            param.requires_grad = trainable
    elif MODEL_NAME == "resnet101":
        for param in model.fc.parameters():
            param.requires_grad = trainable
    elif MODEL_NAME in {"alexnet", "vgg19"}:
        for param in model.classifier.parameters():
            param.requires_grad = trainable
    elif MODEL_NAME == "swin_v2_b":
        for param in model.head.parameters():
            param.requires_grad = trainable
    else:
        raise ValueError(f"Unsupported model name: {MODEL_NAME}")


def set_trainable_layers(model, train_head_only):
    if train_head_only:
        for param in model.parameters():
            param.requires_grad = False
        _set_classifier_trainable(model, trainable=True)
    else:
        for param in model.parameters():
            param.requires_grad = True


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

    set_trainable_layers(model, train_head_only=train_head_only)
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


def build_optimizer(model, learning_rate):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check layer freezing config.")
    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


def get_training_phases():
    if ENABLE_TWO_PHASE_FINETUNE:
        return [
            {
                "name": "head_only",
                "epochs": HEAD_EPOCHS,
                "train_head_only": True,
                "learning_rate": LEARNING_RATE,
            },
            {
                "name": "full_unfreeze",
                "epochs": UNFREEZE_EPOCHS,
                "train_head_only": False,
                "learning_rate": UNFREEZE_LEARNING_RATE,
            },
        ]

    return [
        {
            "name": "single_phase",
            "epochs": EPOCHS,
            "train_head_only": TRAIN_HEAD_ONLY,
            "learning_rate": LEARNING_RATE,
        }
    ]


def train_model_in_phases(model, train_loader, eval_loader, criterion, device):
    history = []

    for phase in get_training_phases():
        set_trainable_layers(model, train_head_only=phase["train_head_only"])
        optimizer = build_optimizer(model, learning_rate=phase["learning_rate"])

        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Starting phase={phase['name']} | epochs={phase['epochs']} | "
            f"lr={phase['learning_rate']} | trainable_params={trainable_count}"
        )

        for epoch in range(phase["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
            eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)

            entry = {
                "phase": phase["name"],
                "epoch": epoch + 1,
                "epochs_in_phase": phase["epochs"],
                "learning_rate": phase["learning_rate"],
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            }
            history.append(entry)
            print(
                f"[{phase['name']}] epoch {epoch + 1}/{phase['epochs']} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}"
            )

    return history


def get_transform(pretrained):
    if pretrained:
        match MODEL_NAME:
            case "vit_b16_head_only":
                return ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            case "alexnet":
                return models.AlexNet_Weights.IMAGENET1K_V1.transforms()
            case "resnet101":
                return models.ResNet101_Weights.IMAGENET1K_V1.transforms()
            case "swin_v2_b":
                return models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms()
            case "vgg19":
                return models.VGG19_Weights.IMAGENET1K_V1.transforms()
            case _:
                raise ValueError(f"Unsupported model name: {MODEL_NAME}")

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)

    data = pd.read_csv(CSV_PATH)
    data = data[["uid", IMAGE_COL, LABEL_COL]].dropna()
    data[IMAGE_COL] = data[IMAGE_COL].astype(str).str.strip()
    data[LABEL_COL] = data[LABEL_COL].astype(str).str.strip()
    data["uid"] = data["uid"].astype(str).str.strip()

    labels = sorted(data[LABEL_COL].unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    train_df, test_df = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    transform = get_transform(PRETRAINED)

    train_records = build_records(train_df, label_to_idx)
    test_records = build_records(test_df, label_to_idx)
    if not train_records or not test_records:
        raise ValueError("No train/test records found. Check CSV_PATH and image locations.")

    train_loader = DataLoader(
        VisionDataset(train_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        VisionDataset(test_records, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_vision_model(PRETRAINED, len(labels), TRAIN_HEAD_ONLY).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"Train: {len(train_records)} | Test: {len(test_records)} | Classes: {len(labels)}")
    print(f"Device: {device}")

    history = train_model_in_phases(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved model weights to: {MODEL_SAVE_PATH}")

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        image_name = str(row[IMAGE_COL]).strip()
        image_path = resolve_image_path(image_name)
        if image_path is None:
            continue

        pred_label, prob_dict = predict_with_probs(model, image_path, transform, idx_to_label, device)
        results.append(
            {
                "id": row["uid"],
                "predicted_diagnosis": pred_label,
                "ground_truth": str(row[LABEL_COL]).strip(),
                "probabilities": prob_dict,
            }
        )

    final_eval_loss, final_eval_acc = evaluate(model, test_loader, criterion, device)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved training history + predictions to: {RESULTS_PATH}")
