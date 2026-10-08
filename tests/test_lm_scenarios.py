"""
Проверка распознавания страницы через LM Studio.

Отправляет изображение локальному серверу LM Studio и сохраняет ответ
в файл. Ничего не уходит в интернет: адрес — localhost.

Запуск:
    python tests/test_lm_scenarios.py <путь к странице> [метка]
"""

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

LM_URL = "http://localhost:1234/v1/chat/completions"
MODELS = [
    "gemma-4-e4b-uncensored-hauhaucs-aggressive",
    "google/gemma-3-12b",
]

# Промпт без привязки к какой-либо организации: только приведение
# документа в читаемый вид.
PROMPT = (
    "Ты распознаватель документов. Извлеки весь текст с этой страницы, "
    "сохраняя структуру: заголовки, абзацы, списки, таблицы. "
    "Верни HTML-разметку текста. Не добавляй пояснений."
)


def ask(image: Image.Image, prompt: str, model: str, timeout: int = 600):
    """
    Отправляет изображение в LM Studio.

    @param image: страница документа.
    @param prompt: инструкция для модели.
    @param model: имя модели.
    @param timeout: предел ожидания в секундах.
    @returns: текст ответа и время в секундах.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": 4000,
        "stream": False,
    }

    started = time.time()
    response = requests.post(LM_URL, json=payload, timeout=timeout)
    elapsed = time.time() - started
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"].get("content", ""), elapsed


def main() -> None:
    """Прогоняет страницу через доступные модели."""
    if len(sys.argv) < 2:
        print("укажите путь к изображению")
        sys.exit(1)

    source = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else source.stem
    out_dir = Path(__file__).resolve().parent.parent / "output" / "lm_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(source).convert("RGB")
    print(f"страница: {image.size[0]}x{image.size[1]}")
    print(f"промпт: {PROMPT[:80]}...\n")

    report = []
    for model in MODELS:
        print(f"--- модель: {model} ---")
        try:
            answer, elapsed = ask(image, PROMPT, model)
            chars = len(answer)
            print(f"    ответ: {chars} символов за {elapsed:.1f} сек")
            (out_dir / f"{label}__{model.replace('/', '_')}.html").write_text(
                answer, encoding="utf-8"
            )
            report.append({
                "model": model,
                "chars": chars,
                "seconds": round(elapsed, 1),
                "ok": chars > 100,
            })
        except Exception as error:
            print(f"    ошибка: {type(error).__name__}: {error}")
            report.append({"model": model, "error": str(error)[:120], "ok": False})
        print()

    (out_dir / f"{label}__report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"результаты: {out_dir}")


if __name__ == "__main__":
    main()
