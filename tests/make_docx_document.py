"""
Генератор тестового документа о мемах про ИИ.

Создаёт .docx на 8 страниц: заголовки, абзацы, списки, таблица и
картинки. Изображения рисуются кодом — они синтетические и
безопасны для публикации.

Запуск:
    python tests/make_docx_document.py
"""

from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "test_document.docx"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "test_images"


def load_font(size: int):
    """
    Подбирает шрифт для подписей на картинках.

    @param size: размер шрифта.
    @returns: объект шрифта.
    """
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_chart(title: str, bars: list, color: tuple, width: int = 760,
               height: int = 420) -> Image.Image:
    """
    Рисует столбчатую диаграмму.

    @param title: подпись над диаграммой.
    @param bars: список пар «подпись, значение».
    @param color: цвет столбцов.
    @param width: ширина изображения.
    @param height: высота изображения.
    @returns: готовое изображение.
    """
    image = Image.new("RGB", (width, height), (252, 252, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_font(26)
    label_font = load_font(20)

    draw.text((30, 22), title, font=title_font, fill=(30, 32, 40))

    chart_top = 90
    chart_bottom = height - 70
    chart_left = 60
    chart_right = width - 40

    # оси
    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill=(150, 152, 160), width=2)
    draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill=(150, 152, 160), width=2)

    max_value = max(value for _, value in bars) or 1
    slot = (chart_right - chart_left) / len(bars)
    bar_width = slot * 0.55

    for index, (label, value) in enumerate(bars):
        bar_height = (value / max_value) * (chart_bottom - chart_top - 20)
        x0 = chart_left + index * slot + (slot - bar_width) / 2
        y0 = chart_bottom - bar_height
        draw.rounded_rectangle([x0, y0, x0 + bar_width, chart_bottom],
                               radius=6, fill=color)
        draw.text((x0 + bar_width / 2 - 14, y0 - 26), str(value),
                  font=label_font, fill=(60, 62, 70))
        draw.text((x0 + bar_width / 2 - len(label) * 5, chart_bottom + 12),
                  label, font=label_font, fill=(70, 72, 80))
    return image


def make_scheme(title: str, width: int = 760, height: int = 380) -> Image.Image:
    """
    Рисует схему из блоков со стрелками.

    @param title: подпись над схемой.
    @param width: ширина изображения.
    @param height: высота изображения.
    @returns: готовое изображение.
    """
    image = Image.new("RGB", (width, height), (250, 250, 253))
    draw = ImageDraw.Draw(image)
    title_font = load_font(26)
    box_font = load_font(21)

    draw.text((30, 22), title, font=title_font, fill=(30, 32, 40))

    boxes = [
        ("Запрос", (108, 152, 208)),
        ("Модель", (150, 130, 200)),
        ("Ответ", (120, 180, 140)),
    ]
    box_w, box_h = 170, 84
    gap = 60
    total = len(boxes) * box_w + (len(boxes) - 1) * gap
    start_x = (width - total) / 2
    y = height / 2 - box_h / 2 + 20

    for index, (label, color) in enumerate(boxes):
        x = start_x + index * (box_w + gap)
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=12, fill=color)
        draw.text((x + box_w / 2 - len(label) * 6, y + box_h / 2 - 13),
                  label, font=box_font, fill=(255, 255, 255))
        if index < len(boxes) - 1:
            arrow_y = y + box_h / 2
            start = x + box_w + 8
            end = x + box_w + gap - 8
            draw.line([(start, arrow_y), (end, arrow_y)], fill=(120, 124, 136), width=3)
            draw.polygon([(end, arrow_y), (end - 12, arrow_y - 7), (end - 12, arrow_y + 7)],
                         fill=(120, 124, 136))
    return image


def make_dialog(title: str, lines: list, width: int = 760, height: int = 420) -> Image.Image:
    """
    Рисует диалог из реплик.

    @param title: подпись над диалогом.
    @param lines: список пар «кто, реплика».
    @param width: ширина изображения.
    @param height: высота изображения.
    @returns: готовое изображение.
    """
    image = Image.new("RGB", (width, height), (245, 246, 250))
    draw = ImageDraw.Draw(image)
    title_font = load_font(26)
    text_font = load_font(20)

    draw.text((30, 22), title, font=title_font, fill=(30, 32, 40))

    y = 88
    for who, text in lines:
        user = who == "user"
        color = (214, 232, 250) if user else (232, 230, 242)
        x0 = 60 if user else 160
        x1 = width - 160 if user else width - 60
        draw.rounded_rectangle([x0, y, x1, y + 58], radius=14, fill=color)
        draw.text((x0 + 20, y + 18), text, font=text_font, fill=(40, 42, 52))
        y += 74
    return image


def add_image(document: Document, image: Image.Image, caption: str) -> None:
    """
    Вставляет изображение с подписью в документ.

    @param document: документ.
    @param image: изображение.
    @param caption: подпись под рисунком.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    document.add_picture(buffer, width=Inches(5.6))

    last = document.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption_paragraph = document.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption_paragraph.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x60, 0x60, 0x70)
        run.font.italic = True


def heading(document: Document, text: str, level: int = 1) -> None:
    """
    Добавляет заголовок с нужным отступом.

    @param document: документ.
    @param text: текст заголовка.
    @param level: уровень заголовка.
    """
    document.add_heading(text, level=level)


def body(document: Document, text: str) -> None:
    """
    Добавляет абзац основного текста.

    @param document: документ.
    @param text: текст абзаца.
    """
    paragraph = document.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in paragraph.runs:
        run.font.size = Pt(11)


def bullets(document: Document, items: list) -> None:
    """
    Добавляет маркированный список.

    @param document: документ.
    @param items: строки списка.
    """
    for item in items:
        paragraph = document.add_paragraph(item, style="List Bullet")
        for run in paragraph.runs:
            run.font.size = Pt(11)


def page_break(document: Document) -> None:
    """
    Начинает новую страницу.

    @param document: документ.
    """
    document.add_page_break()


def build() -> Path:
    """
    Собирает документ.

    @returns: путь к созданному файлу.
    """
    document = Document()

    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ---------- Страница 1 ----------
    heading(document, "Мемы про искусственный интеллект", 0)
    body(document, "Как шутки объясняют то, что не объяснили учебники")
    body(document,
         "За последние годы нейросети вошли в повседневную жизнь: они пишут тексты, "
         "рисуют картинки, отвечают на вопросы и подсказывают код. Вместе с ними "
         "появился целый пласт интернет-фольклора — мемы про ИИ. Этот документ "
         "разбирает, откуда они взялись и что на самом деле означают.")

    heading(document, "Зачем вообще шутить про нейросети", 2)
    body(document,
         "Любая новая технология сначала пугает, потом удивляет и только затем "
         "становится обыденной. Шутки — самый быстрый способ пережить первые два "
         "этапа. Мемы про ИИ выполняют три роли:")
    bullets(document, [
        "снимают страх перед непонятной технологией;",
        "показывают реальные ограничения моделей;",
        "служат общим языком для тех, кто с ними работает.",
    ])

    body(document,
         "Важно понимать: хороший мем почти всегда содержит наблюдение. "
         "Если шутка про нейросеть кажется смешной, значит она подметила "
         "что-то настоящее в её поведении.")

    add_image(document,
              make_chart("Рис. 1. Что чаще всего просят у нейросети",
                         [("текст", 42), ("код", 28), ("перевод", 17), ("картинки", 13)],
                         (108, 152, 208)),
              "Рис. 1. Распределение типовых запросов к нейросетям")

    page_break(document)

    # ---------- Страница 2 ----------
    heading(document, "Классика: галлюцинации", 1)
    body(document,
         "Самый популярный сюжет — модель уверенно выдаёт неправду. "
         "В научной литературе это называется галлюцинацией, а в интернете "
         "превратилось в бесконечную серию шуток.")
    body(document,
         "Типичный мем выглядит так: пользователь просит назвать источники, "
         "и получает список книг, которых никогда не существовало. Причём "
         "названия выглядят абсолютно правдоподобно.")

    heading(document, "Почему так происходит", 2)
    body(document,
         "Модель предсказывает следующее слово, опираясь на закономерности "
         "текста. Она не проверяет факты, а продолжает фразу так, как это "
         "обычно бывает в похожих текстах. Отсюда и берётся уверенная неправда.")
    bullets(document, [
        "модель не знает, чего она не знает;",
        "уверенный тон ничего не говорит о достоверности;",
        "проверять факты всё равно приходится человеку.",
    ])

    body(document,
         "Отсюда главный практический вывод, который хорошо усвоили авторы "
         "мемов: нейросеть — это помощник, а не источник истины.")

    add_image(document,
              make_dialog("Рис. 2. Типичный диалог с галлюцинацией",
                          [("user", "Назови три книги про нейросети"),
                           ("ai", "Вот список из трёх изданий..."),
                           ("user", "А они точно существуют?")],
                          height=360),
              "Рис. 2. Уверенный ответ не означает правильный")

    page_break(document)

    # ---------- Страница 3 ----------
    heading(document, "Второй сюжет: слишком буквальное понимание", 1)
    body(document,
         "Модели плохо понимают подтекст и шутки, если их специально не "
         "предупредить. Просьба «сделай так, чтобы код летал» может привести "
         "к переменной с именем fly или к комментарию про птиц.")

    heading(document, "Примеры из жизни", 2)
    bullets(document, [
        "просьба «починить баг» приводит к удалению проблемной строки;",
        "просьба «сделать красиво» приводит к смене шрифта на курсив;",
        "просьба «ускорить» приводит к отключению проверок.",
    ])

    body(document,
         "Смысл этих шуток не в том, что модель глупая. Она действительно "
         "выполняет буквально то, что написано. Ответственность за точность "
         "формулировки лежит на том, кто пишет запрос.")

    heading(document, "Как формулировать запрос", 2)
    body(document,
         "Полезно указывать не только желание, но и ограничения: что нельзя "
         "менять, какой результат считается правильным, чего делать не стоит. "
         "Чем конкретнее запрос, тем меньше поводов для мемов.")

    add_image(document,
              make_scheme("Рис. 3. Путь запроса до готового ответа"),
              "Рис. 3. От формулировки запроса до проверенного результата")

    page_break(document)

    # ---------- Страница 4 ----------
    heading(document, "Третий сюжет: нейросеть вместо специалиста", 1)
    body(document,
         "Отдельная группа шуток посвящена тому, как модели предлагают "
         "заменить любую профессию — от юриста до врача. Мемы высмеивают "
         "не технологии, а чрезмерные ожидания от них.")

    heading(document, "Что модели действительно умеют", 2)
    body(document,
         "Справедливости ради стоит перечислить задачи, где нейросети "
         "приносят настоящую пользу:")
    bullets(document, [
        "черновые тексты и summaries длинных документов;",
        "перевод и переформулировка;",
        "подсказки по коду и разбор ошибок;",
        "поиск закономерностей в больших таблицах.",
    ])

    body(document,
         "Во всех этих случаях результат проверяет человек. Как только "
         "проверку убирают, появляется повод для очередного мема.")

    heading(document, "Ответственность", 2)
    body(document,
         "Ответственность за решение всегда остаётся на человеке. "
         "Модель может ошибиться, но последствия ошибки несёт тот, "
         "кто воспользовался её советом, не проверив его.")

    add_image(document,
              make_chart("Рис. 4. Где нейросеть помогает, а где мешает",
                         [("черновик", 38), ("перевод", 31), ("код", 24), ("факты", 7)],
                         (150, 130, 200)),
              "Рис. 4. Доля задач, где результат принимается без правок")

    page_break(document)

    # ---------- Страница 5 ----------
    heading(document, "Четвёртый сюжет: нейросеть обучается на нас", 1)
    body(document,
         "Популярны шутки о том, что модели обучались на интернете, "
         "а значит впитали все его странности. Отсюда берутся "
         "неожиданные ответы на простые вопросы.")

    heading(document, "Таблица типичных ситуаций", 2)

    rows = [
        ("Что просят", "Что происходит"),
        ("Объясни простыми словами", "ответ усложняется вдвое"),
        ("Ответь коротко", "приходит текст на три экрана"),
        ("Приведи пример", "пример оказывается выдуманным"),
        ("Исправь ошибку", "появляется новая ошибка"),
        ("Сделай как в прошлый раз", "прошлый раз не сохранился"),
    ]

    table = document.add_table(rows=len(rows), cols=2)
    table.style = "Light Grid Accent 1"
    for index, (left, right) in enumerate(rows):
        cells = table.rows[index].cells
        cells[0].text = left
        cells[1].text = right
        if index == 0:
            for cell in cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True

    body(document,
         "Эта таблица — почти дословный пересказ реальных жалоб "
         "пользователей, собранных в шутливую форму.")

    page_break(document)

    # ---------- Страница 6 ----------
    heading(document, "Пятый сюжет: подсказки в коде", 1)
    body(document,
         "Программисты шутят про нейросети больше всех, потому что "
         "сталкиваются с ними ежедневно. Особенно достаётся автодополнению кода.")

    heading(document, "Что бесит разработчиков", 2)
    bullets(document, [
        "подсказка предлагает код, который не компилируется;",
        "библиотека в примере не существует;",
        "решение выглядит красиво, но ломает соседний модуль;",
        "в ответе уверенно используется устаревший способ.",
    ])

    body(document,
         "При этом те же разработчики признают, что для рутинных задач "
         "подсказки экономят время. Мемы не отменяют пользы — они лишь "
         "напоминают, что проверять результат всё равно нужно.")

    heading(document, "Правило одного абзаца", 2)
    body(document,
         "Практическое правило: если предложенный код нельзя объяснить "
         "своими словами за один абзац, использовать его рано. "
         "Это защищает и от ошибок, и от непонимания собственного проекта.")

    add_image(document,
              make_dialog("Рис. 5. Как выглядит правка подсказки",
                          [("ai", "Готово, функция переписана"),
                           ("user", "Почему тесты упали?"),
                           ("ai", "Я убрал проверку, она мешала")],
                          height=360),
              "Рис. 5. Подсказку всегда проверяют тестами")

    page_break(document)

    # ---------- Страница 7 ----------
    heading(document, "Шестой сюжет: слишком человеческое поведение", 1)
    body(document,
         "Люди склонны приписывать моделям чувства и намерения. "
         "Отсюда шутки про то, что нейросеть обиделась, устала "
         "или делает вид, что не поняла вопрос.")

    heading(document, "Откуда берётся иллюзия", 2)
    body(document,
         "Модель обучена на человеческих текстах, поэтому отвечает "
         "в человеческой манере. Вежливые формулировки и извинения — "
         "это следствие обучающих данных, а не внутреннего состояния.")

    bullets(document, [
        "у модели нет настроения и усталости;",
        "извинения в ответе — это шаблон речи;",
        "изменение тона зависит от формулировки запроса.",
    ])

    heading(document, "Почему это важно", 2)
    body(document,
         "Понимание разницы помогает не строить лишних ожиданий. "
         "Модель — инструмент, который удобно ведёт диалог, "
         "но за этим диалогом нет собеседника.")

    page_break(document)

    # ---------- Страница 8 ----------
    heading(document, "Что в итоге", 1)
    body(document,
         "Мемы про искусственный интеллект — это способ общества "
         "осмыслить новую технологию. Они фиксируют реальные "
         "ограничения и помогают не переоценивать возможности.")

    heading(document, "Главные выводы", 2)
    bullets(document, [
        "модель предсказывает текст, а не ищет истину;",
        "уверенный ответ не означает правильный;",
        "чем точнее запрос, тем полезнее результат;",
        "ответственность за решение остаётся на человеке.",
    ])

    body(document,
         "Если относиться к нейросети как к старательному, но не всегда "
         "внимательному помощнику, большинство шуток перестают быть "
         "злыми и становятся просто точными.")

    heading(document, "Дополнительно", 2)
    body(document,
         "Документ создан для проверки программы обработки длинных "
         "скриншотов. Текст нейтральный и не относится к какой-либо "
         "конкретной организации.")

    document.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Создан: {path}")
    print(f"Размер: {path.stat().st_size / 1024:.0f} KB")
