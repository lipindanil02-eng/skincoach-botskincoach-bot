"""
inference.py — Предсказание кожного заболевания по фото
Подключается к боту как модуль
"""
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "best_model.pth"
CLASS_MAP_PATH = "class_map.json"
IMG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.5

CLASS_LABELS_RU = {
    "psoriasis":   "Псориаз",
    "dermatitis":  "Дерматит",
    "eczema":      "Экзема",
    "melanoma":    "Меланома",
    "nevus":       "Невус (родинка)",
    "other":       "Другое заболевание",
}

_model = None
_class_map = None
_device = None

def load_model():
    global _model, _class_map, _device
    if _model is not None:
        return

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        _class_map = json.load(f)

    num_classes = len(_class_map)
    checkpoint = torch.load(MODEL_PATH, map_location=_device)
    model_name = checkpoint.get("model_name", "efficientnet_b3")

    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    else:
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    _model = model
    print(f"✅ Модель загружена | Классов: {num_classes} | {_device}")


_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path: str) -> dict:
    """
    Предсказать заболевание по фото.
    Возвращает: diagnosis, diagnosis_ru, confidence, top3, reliable, message
    """
    load_model()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Не удалось открыть изображение: {e}"}

    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_k = min(3, len(_class_map))
    top_probs, top_idx = probs.topk(top_k)
    top_probs = top_probs.cpu().numpy()
    top_idx = top_idx.cpu().numpy()

    # Определяем маппинг индекс → класс
    first_key = list(_class_map.keys())[0]
    if str(first_key).isdigit():
        idx_to_class = {int(k): v for k, v in _class_map.items()}
    else:
        idx_to_class = {v: k for k, v in _class_map.items()}

    best_class = idx_to_class.get(int(top_idx[0]), f"class_{top_idx[0]}")
    best_prob = float(top_probs[0])
    diagnosis_ru = CLASS_LABELS_RU.get(best_class, best_class)
    reliable = best_prob >= CONFIDENCE_THRESHOLD

    top3 = []
    for idx, prob in zip(top_idx, top_probs):
        cls = idx_to_class.get(int(idx), f"class_{idx}")
        top3.append({
            "diagnosis": cls,
            "diagnosis_ru": CLASS_LABELS_RU.get(cls, cls),
            "confidence_pct": f"{float(prob)*100:.1f}%"
        })

    if reliable:
        message = f"🔬 Модель определила: {diagnosis_ru} (уверенность {best_prob*100:.1f}%)"
    else:
        message = f"🔬 Вероятнее всего: {diagnosis_ru} ({best_prob*100:.1f}% — требует уточнения)"

    return {
        "diagnosis": best_class,
        "diagnosis_ru": diagnosis_ru,
        "confidence": round(best_prob, 4),
        "confidence_pct": f"{best_prob*100:.1f}%",
        "top3": top3,
        "reliable": reliable,
        "message": message,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = predict_image(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Использование: python inference.py path/to/image.jpg")
