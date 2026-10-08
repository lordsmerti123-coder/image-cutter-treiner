"""
Проверка детекции страниц на тестовом документе.

Загружает обученную модель, прогоняет тестовый скриншот и печатает
найденные рамки страниц. Модель и скрипт не изменяются — только чтение.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_cutter_model.pth"
IMAGE_PATH = Path(__file__).resolve().parent.parent / "test_long_document.png"
MAX_SIDE = 1024


def load_client_model(model_path: Path, device: torch.device):
    """Собирает ту же архитектуру, что использует рабочий скрипт."""
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def detect(model, image: Image.Image, device: torch.device, confidence: float = 0.5):
    """Возвращает рамки страниц в координатах исходного изображения."""
    width, height = image.size
    scale = MAX_SIDE / max(width, height) if max(width, height) > MAX_SIDE else 1.0
    small = image.resize((int(width * scale), int(height * scale)), Image.BICUBIC) if scale != 1.0 else image

    tensor = T.ToTensor()(small).to(device)
    with torch.no_grad():
        prediction = model([tensor])

    boxes = prediction[0]["boxes"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()
    found = [
        (int(b[0] / scale), int(b[1] / scale), int(b[2] / scale), int(b[3] / scale), float(s))
        for b, s in zip(boxes, scores)
        if s >= confidence
    ]
    found.sort(key=lambda item: item[1])
    return found


def main() -> None:
    """Прогоняет детекцию и печатает результат."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    if not MODEL_PATH.exists():
        print(f"Модель не найдена: {MODEL_PATH}")
        sys.exit(2)

    print(f"Загрузка модели: {MODEL_PATH.name}")
    model = load_client_model(MODEL_PATH, device)

    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Изображение: {image.size[0]}x{image.size[1]}")

    for confidence in (0.5, 0.3, 0.1):
        boxes = detect(model, image, device, confidence)
        print(f"\nпорог {confidence}: найдено {len(boxes)} страниц")
        for index, (x1, y1, x2, y2, score) in enumerate(boxes, 1):
            print(f"  {index}. y={y1}..{y2} (высота {y2 - y1}), x={x1}..{x2}, "
                  f"уверенность {score:.3f}")


if __name__ == "__main__":
    main()
