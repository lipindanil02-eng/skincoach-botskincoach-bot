"""
train.py — Обучение модели классификации кожных заболеваний
Совместим с Google Colab (T4 GPU, 15GB VRAM)
Базовая модель: EfficientNet-B3 (предобученная на ImageNet)

Классы: псориаз, дерматит/экзема, меланома, невус, базально-клеточная карцинома,
        актинический кератоз, сосудистые поражения

Использование:
  python train.py --data_dir ./dataset --epochs 20 --batch_size 32
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ─────────────────────────────────────────────
# АРГУМЕНТЫ
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Skin Disease Classifier Training")
    parser.add_argument("--data_dir",    type=str, default="./dataset",
                        help="Путь к папке с данными (train/val/test)")
    parser.add_argument("--output_dir",  type=str, default="./output",
                        help="Куда сохранять веса и логи")
    parser.add_argument("--model",       type=str, default="efficientnet_b3",
                        choices=["efficientnet_b3", "resnet50", "convnext_small"],
                        help="Базовая архитектура")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--img_size",    type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_base", action="store_true",
                        help="Заморозить базовую модель (только обучать голову)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# ТРАНСФОРМАЦИИ
# ─────────────────────────────────────────────
def get_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────
def build_model(model_name: str, num_classes: int, freeze_base: bool):
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights="IMAGENET1K_V1")
        if freeze_base:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2")
        if freeze_base:
            for name, p in model.named_parameters():
                if "layer4" not in name and "fc" not in name:
                    p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "convnext_small":
        model = models.convnext_small(weights="IMAGENET1K_V1")
        if freeze_base:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    return model


# ─────────────────────────────────────────────
# БАЛАНСИРОВКА КЛАССОВ (WeightedSampler)
# ─────────────────────────────────────────────
def make_weighted_sampler(dataset):
    class_counts = np.bincount(dataset.targets)
    weights = 1.0 / class_counts
    sample_weights = [weights[t] for t in dataset.targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


# ─────────────────────────────────────────────
# ОБУЧЕНИЕ ОДНОЙ ЭПОХИ
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# ВАЛИДАЦИЯ
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Устройство: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Данные
    train_tf, val_tf = get_transforms(args.img_size)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"📂 Классов: {num_classes} → {train_ds.classes}")
    print(f"📸 Обучение: {len(train_ds)} | Валидация: {len(val_ds)}")

    # Сохраняем маппинг классов
    class_map = {v: k for k, v in train_ds.class_to_idx.items()}
    with open(output_path / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # Модель
    model = build_model(args.model, num_classes, args.freeze_base).to(device)
    print(f"🧠 Модель: {args.model} | Заморозка базы: {args.freeze_base}")

    # Потери — с учётом дисбаланса классов
    class_counts = np.bincount(train_ds.targets)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Обучение
    best_val_acc = 0.0
    history = []

    print("\n" + "="*60)
    print("🚀 Начало обучения")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Эпоха {epoch:03d}/{args.epochs} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"⏱ {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        })

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_ds.classes,
                "model_name": args.model,
                "img_size": args.img_size,
            }, output_path / "best_model.pth")
            print(f"  ✅ Лучшая модель сохранена (val_acc={val_acc:.4f})")

    # Финальный отчёт
    print("\n" + "="*60)
    print("📊 Финальный отчёт (на валидационной выборке)")
    print("="*60)
    print(classification_report(labels, preds, target_names=train_ds.classes))

    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Готово! Лучшая точность: {best_val_acc:.4f}")
    print(f"📁 Файлы сохранены в: {output_path}")
    print(f"   - best_model.pth  ← веса модели")
    print(f"   - class_map.json  ← маппинг классов")
    print(f"   - history.json    ← история обучения")


if __name__ == "__main__":
    main()
