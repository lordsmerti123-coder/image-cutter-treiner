"""
Проверка нарезки длинного скриншота на страницы.

Повторяет логику рабочего скрипта без графического интерфейса:
детекция страниц моделью, вырезание вложенных картинок, сохранение
результата на диск.

Запуск:
    python tests/test_split.py test_long_document.png
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_NAME = "trained_cutter_model.pth"
MAX_SIDE = 1024
# Страница — это цельный блок документа. Модель помечает страницы
# уверенностью 0.9 и выше; картинки внутри страницы и отдельные абзацы
# попадают в диапазон 0.2-0.8. Разделять по уверенности надёжнее, чем
# по ширине рамки: у мелких блоков ширина почти такая же, как у страницы.
CONFIDENCE = 0.9
# Картинки внутри страницы модель помечает уверенностью 0.5-0.8,
# поэтому для них используется отдельный, более низкий порог.
NESTED_CONFIDENCE = 0.45


def load_client_model(model_path: Path, device: torch.device):
    """
    Собирает архитектуру детектора и загружает обученные веса.

    @param model_path: путь к файлу .pth.
    @param device: устройство вычислений.
    @returns: модель в режиме оценки.
    """
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def detect_boxes(model, image: Image.Image, device: torch.device,
                 confidence: float | None = None):
    """
    Находит рамки страниц и картинок на изображении.

    @param model: загруженный детектор.
    @param image: исходный скриншот.
    @param device: устройство вычислений.
    @param confidence: порог уверенности; по умолчанию CONFIDENCE.
    @returns: список словарей с рамкой и уверенностью, сверху вниз.
    """
    threshold = CONFIDENCE if confidence is None else confidence
    width, height = image.size
    scale = MAX_SIDE / max(width, height) if max(width, height) > MAX_SIDE else 1.0
    small = image.resize((int(width * scale), int(height * scale)), Image.BICUBIC) if scale != 1.0 else image

    with torch.no_grad():
        prediction = model([T.ToTensor()(small).to(device)])

    boxes = prediction[0]["boxes"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()
    found = [
        {"box": tuple(int(c / scale) for c in b), "score": float(s)}
        for b, s in zip(boxes, scores)
        if s >= threshold
    ]
    found.sort(key=lambda item: item["box"][1])
    return found


def split_pages(boxes, image: Image.Image):
    """
    Отделяет страницы от картинок внутри них.

    Страницы приходят с высоким порогом уверенности. Картинки ищутся
    отдельным проходом с низким порогом и распределяются по страницам
    по своему положению.

    @param boxes: рамки страниц, найденные с высоким порогом.
    @param image: исходный скриншот.
    @returns: список страниц с вложенными в них картинками.
    """
    pages = [
        {"box": item["box"], "score": item["score"], "images": []}
        for item in _merge_overlapping(boxes)
    ]
    if not pages:
        return pages

    pages.sort(key=lambda page: page["box"][1])
    return pages


def _merge_overlapping(boxes):
    """
    Убирает рамки, которые перекрывают уже принятые.

    Модель иногда даёт две почти одинаковые рамки на одну страницу.
    Оставляем ту, у которой выше уверенность.

    @param boxes: найденные рамки, отсортированные сверху вниз.
    @returns: список неперекрывающихся рамок.
    """
    kept = []
    for item in sorted(boxes, key=lambda entry: -entry["score"]):
        _, y1, _, y2 = item["box"]
        overlaps = False
        for page in kept:
            _, py1, _, py2 = page["box"]
            intersection = min(y2, py2) - max(y1, py1)
            if intersection > 0.5 * min(y2 - y1, py2 - py1):
                overlaps = True
                break
        if not overlaps:
            kept.append(item)
    return kept


def main() -> None:
    """Запускает нарезку тестового документа и сохраняет результат."""
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "test_long_document.png")
    # Результат идёт рядом с папкой проекта, независимо от текущего каталога.
    out_dir = Path(__file__).resolve().parent.parent / "output" / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).resolve().parent.parent / "models" / MODEL_NAME
    print(f"Устройство: {device}")
    print(f"Модель: {model_path}")
    if not model_path.exists():
        print("Модель не найдена")
        sys.exit(2)

    model = load_client_model(model_path, device)
    image = Image.open(source).convert("RGB")
    print(f"Документ: {image.size[0]}x{image.size[1]}")

    # Страницы — высокий порог, картинки внутри — низкий.
    page_boxes = detect_boxes(model, image, device, CONFIDENCE)
    nested_boxes = detect_boxes(model, image, device, NESTED_CONFIDENCE)

    pages = split_pages(page_boxes, image)
    for box in nested_boxes:
        x1, y1, x2, y2 = box["box"]
        for page in pages:
            _, py1, _, py2 = page["box"]
            if py1 <= y1 and y2 <= py2:
                page["images"].append(box["box"])
                break

    # Модель иногда повторяет рамку страницы как «картинку». Если размеры
    # почти совпадают со страницей, это дубль, а не изображение внутри неё.
    for page in pages:
        px1, py1, px2, py2 = page["box"]
        page["images"] = [
            box for box in page["images"]
            if (box[2] - box[0]) * (box[3] - box[1]) < 0.8 * (px2 - px1) * (py2 - py1)
        ]

    print(f"\nНайдено страниц: {len(pages)}")

    for index, page in enumerate(pages, 1):
        x1, y1, x2, y2 = page["box"]
        crop = image.crop((x1, y1, x2, y2))
        crop.save(out_dir / f"page_{index}.png")
        print(f"  Страница {index}: y={y1}..{y2}, размер {crop.size[0]}x{crop.size[1]}, "
              f"картинок внутри: {len(page['images'])}, уверенность {page['score']:.3f}")

        for image_index, (ix1, iy1, ix2, iy2) in enumerate(page["images"], 1):
            nested = image.crop((ix1, iy1, ix2, iy2))
            nested.save(out_dir / f"page_{index}_img_{image_index}.png")
            print(f"      картинка {image_index}: {nested.size[0]}x{nested.size[1]}")

    print(f"\nРезультат: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
