"""
Проверка файла модели перед публикацией.

Показывает структуру файла и ищет в нём текстовые метаданные, которые
могли бы выдать источник данных. Веса читаются с диска локально,
никуда не отправляются.
"""

from pathlib import Path

import torch

MODEL = Path(__file__).resolve().parent.parent / "models" / "trained_cutter_model.pth"


def scan(obj, path: str = "", found: list | None = None) -> list:
    """
    Рекурсивно ищет строки внутри структуры модели.

    @param obj: проверяемый объект.
    @param path: путь до объекта для отчёта.
    @param found: накопитель найденных строк.
    @returns: список пар «путь, строка».
    """
    if found is None:
        found = []
    if isinstance(obj, str):
        found.append((path, obj[:200]))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            scan(value, f"{path}.{key}", found)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            scan(value, f"{path}[{index}]", found)
    return found


def main() -> None:
    """Печатает структуру модели и найденные строки."""
    size_mb = MODEL.stat().st_size / 1024 / 1024
    print(f"файл: {MODEL.name}")
    print(f"размер: {size_mb:.1f} MB")

    data = torch.load(MODEL, map_location="cpu", weights_only=False)
    print(f"тип: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"ключей верхнего уровня: {len(data)}")
        for key in list(data.keys())[:10]:
            value = data[key]
            shape = getattr(value, "shape", None)
            print(f"   {key}: {type(value).__name__}"
                  + (f" {tuple(shape)}" if shape is not None else ""))

    print("\nпоиск текстовых строк внутри файла:")
    strings = scan(data)
    if not strings:
        print("   строк не найдено — метаданных нет")
    else:
        for path, text in strings:
            print(f"   {path}: {text!r}")

    print(f"\nвсего строк: {len(strings)}")


if __name__ == "__main__":
    main()
