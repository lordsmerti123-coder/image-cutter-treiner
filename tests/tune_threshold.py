"""
Подбор порога уверенности под конкретный документ.

Показывает, сколько страниц находится при разных порогах. Помогает
понять, подходит ли модель для данного типа документа.

Запуск:
    python tests/tune_threshold.py [путь к изображению]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_split
from test_split import detect_boxes, load_client_model, split_pages

import torch
from PIL import Image


def main() -> None:
    """Печатает число найденных страниц при разных порогах."""
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "test_long_document.png"
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_client_model(
        Path(__file__).resolve().parent.parent / "models" / "trained_cutter_model.pth",
        device,
    )
    image = Image.open(source).convert("RGB")
    print(f"документ: {image.size[0]}x{image.size[1]}")
    if expected:
        print(f"ожидается страниц: {expected}")
    print()

    for confidence in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        boxes = detect_boxes(model, image, device, confidence)
        pages = split_pages(boxes, image)
        heights = [page["box"][3] - page["box"][1] for page in pages]
        mark = ""
        if expected and len(pages) == expected:
            mark = "  <-- совпадает с ожидаемым"
        print(f"порог {confidence}: рамок {len(boxes):3}, страниц {len(pages):3}"
              f"  высоты: {heights[:10]}{mark}")


if __name__ == "__main__":
    main()
