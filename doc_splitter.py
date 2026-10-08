"""
Программа для обработки длинных скриншотов: детекция страниц, вырезание вложенных
изображений и создание HTML с распознанным текстом (LM Studio / без текста).
Поддерживает паузу, остановку, выбор отдельных файлов из папки.
"""
import os, base64, threading, re, time, tkinter as tk
from tkinter import filedialog, messagebox, ttk
from io import BytesIO
import requests
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Каталог программы: рядом лежат models/ и output/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_LM_MODEL = "google/gemma-3-12b"
MAX_IMAGE_SIDE = 1024
DEFAULT_CONFIDENCE = 0.5
DEFAULT_MAX_TOKENS = 6000
DEFAULT_TIMEOUT = 300
TITLE_PROMPT_DEFAULT = "Какое название этого документа? Верни только название, без кавычек."

# Дефолтные промпты для двух стилей обработки
PROMPT_FULL_DEFAULT = (
    "Ты распознаватель документов. Не раздумывай. Извлеки весь текст с этой страницы, "
    "сохраняя структуру, цвет, стиль, шрифт, заголовки и списки. Не описывай картинки, только текст. "
    "Верни HTML. Вставь изображения как ссылки с помощью тега <img src=\"имя_файла\">."
)
PROMPT_OCR_DEFAULT = (
    "Ты OCR для документов. Извлеки весь текст с этой страницы, сохраняя абзацы. "
    "Если встречаешь рисунок, вставь тег <img src=\"имя_файла\">. "
    "Не описывай картинки. Верни только HTML."
)


def get_safe_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.tensor([1.0], device='cuda')
        return torch.device('cuda')
    except:
        return torch.device('cpu')


def load_model(model_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(pil_image, max_side=MAX_IMAGE_SIDE):
    w, h = pil_image.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
    img_tensor = T.ToTensor()(pil_image)
    return img_tensor, scale


def predict_boxes(model, pil_image, device, confidence=0.5):
    img_tensor, scale_inv = preprocess_image(pil_image)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        preds = model([img_tensor])
    boxes_np = preds[0]['boxes'].cpu().numpy()
    scores_np = preds[0]['scores'].cpu().numpy()
    results = []
    for b, s in zip(boxes_np, scores_np):
        if s >= confidence:
            orig_box = [int(coord / scale_inv) for coord in b]
            results.append((*orig_box, float(s)))
    results.sort(key=lambda b: b[1])
    return results


def ask_lm_studio(image: Image.Image, prompt: str, url, model_name, max_tokens=500, timeout=60):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
        ]
    }]
    payload = {"messages": messages, "temperature": 0.1, "max_tokens": max_tokens, "stream": False}
    if model_name:
        payload["model"] = model_name
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - start
        print(f"[LM Studio] Запрос занял {elapsed:.1f} сек")
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message'].get('content', '').strip()
        return content if content else None, elapsed
    except requests.exceptions.Timeout:
        print(f"[LM Studio] Тайм-аут ({timeout} сек)")
        return None, timeout
    except Exception as e:
        print(f"[LM Studio Error] {e}")
        return None, 0.0


def clean_filename(name, max_len=50):
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, '_')
    name = name.replace('\n', ' ').replace('\r', '')
    return ' '.join(name.split())[:max_len].strip()


class ReconstructionTab(ttk.Frame):
    def __init__(self, parent, log_func, root):
        super().__init__(parent)
        self.log_func = log_func
        self.root = root
        self.device = torch.device('cpu')
        self.model = None
        self.paused = threading.Event()
        self.paused.set()
        self.stopped = threading.Event()
        self.worker = None

        # Переменные интерфейса
        self.source_mode = tk.StringVar(value="file")
        self.input_path = tk.StringVar()
        self.selected_files = []
        self.output_dir = tk.StringVar(value=os.path.join(BASE_DIR, "output"))
        self.doc_name_manual = tk.StringVar()
        self.title_prompt = tk.StringVar(value=TITLE_PROMPT_DEFAULT)
        self.prompt_var = tk.StringVar()
        self.max_tokens_var = tk.IntVar(value=DEFAULT_MAX_TOKENS)
        self.timeout_var = tk.IntVar(value=DEFAULT_TIMEOUT)
        self.recognition_mode = tk.StringVar(value="lm")
        self.confidence_pages = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.confidence_nested = tk.DoubleVar(value=0.25)
        self.save_images_files = tk.BooleanVar(value=True)
        self.embed_images = tk.BooleanVar(value=True)
        self.find_pages = tk.BooleanVar(value=True)
        self.find_nested = tk.BooleanVar(value=True)
        self.processor_mode = tk.StringVar(value="full")  # full или ocr

        self.last_prompt_full = PROMPT_FULL_DEFAULT
        self.last_prompt_ocr = PROMPT_OCR_DEFAULT

        self.setup_ui()

    def setup_ui(self):
        # Верхняя часть – источник, выход, детекция
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        # Источник
        src_frame = ttk.LabelFrame(top_frame, text="Источник", padding=5)
        src_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(src_frame, text="Файл", variable=self.source_mode,
                        value="file", command=self.toggle_source).grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(src_frame, text="Папка", variable=self.source_mode,
                        value="folder", command=self.toggle_source).grid(row=0, column=1, sticky='w')
        self.path_entry = ttk.Entry(src_frame, textvariable=self.input_path, width=70)
        self.path_entry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5)
        ttk.Button(src_frame, text="Обзор", command=self.browse).grid(row=1, column=2, padx=5)
        ttk.Button(src_frame, text="Выбрать файлы...", command=self.select_files).grid(row=1, column=3, padx=5)

        # Выходная папка
        out_frame = ttk.Frame(top_frame)
        out_frame.pack(fill=tk.X, pady=2)
        ttk.Label(out_frame, text="Выход:").pack(side=tk.LEFT)
        ttk.Entry(out_frame, textvariable=self.output_dir, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(out_frame, text="Обзор", command=lambda: self.output_dir.set(filedialog.askdirectory())).pack(side=tk.LEFT)

        # Детекция
        det_frame = ttk.LabelFrame(self, text="Детекция", padding=5)
        det_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Checkbutton(det_frame, text="Страницы", variable=self.find_pages,
                        command=self.toggle_pages).grid(row=0, column=0, sticky='w')
        self.pages_frame = ttk.Frame(det_frame)
        self.pages_frame.grid(row=0, column=1, padx=10, sticky='w')
        ttk.Label(self.pages_frame, text="Порог:").pack(side=tk.LEFT)
        ttk.Scale(self.pages_frame, from_=0.1, to=0.9, variable=self.confidence_pages, length=100).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(det_frame, text="Вложенные", variable=self.find_nested,
                        command=self.toggle_nested).grid(row=1, column=0, sticky='w')
        self.nested_frame = ttk.Frame(det_frame)
        self.nested_frame.grid(row=1, column=1, padx=10, pady=2, sticky='w')
        ttk.Label(self.nested_frame, text="Порог:").pack(side=tk.LEFT)
        ttk.Scale(self.nested_frame, from_=0.1, to=0.9, variable=self.confidence_nested, length=100).pack(side=tk.LEFT, padx=5)

        # Режим распознавания
        rec_frame = ttk.LabelFrame(self, text="Распознавание", padding=5)
        rec_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Radiobutton(rec_frame, text="LM Studio", variable=self.recognition_mode,
                        value="lm", command=self.toggle_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(rec_frame, text="Без текста", variable=self.recognition_mode,
                        value="none", command=self.toggle_mode).pack(side=tk.LEFT)
        ttk.Label(rec_frame, text="Стиль:").pack(side=tk.LEFT, padx=(20,0))
        ttk.Radiobutton(rec_frame, text="Полный HTML", variable=self.processor_mode,
                        value="full", command=self._on_prompt_mode_changed).pack(side=tk.LEFT)
        ttk.Radiobutton(rec_frame, text="OCR (текст)", variable=self.processor_mode,
                        value="ocr", command=self._on_prompt_mode_changed).pack(side=tk.LEFT)

        # LM Studio настройки
        self.lm_frame = ttk.LabelFrame(self, text="LM Studio", padding=5)
        self.lm_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(self.lm_frame, text="Промпт (редактируемый):").pack(anchor='w')
        self.prompt_text = tk.Text(self.lm_frame, height=6, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.X, padx=5, pady=2)
        param_f = ttk.Frame(self.lm_frame)
        param_f.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(param_f, text="Токенов:").pack(side=tk.LEFT)
        ttk.Entry(param_f, textvariable=self.max_tokens_var, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(param_f, text="Таймаут (сек):").pack(side=tk.LEFT)
        ttk.Entry(param_f, textvariable=self.timeout_var, width=6).pack(side=tk.LEFT, padx=5)

        # Название
        name_frame = ttk.LabelFrame(self, text="Название документа", padding=5)
        name_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(name_frame, text="Вручную:").pack(side=tk.LEFT)
        ttk.Entry(name_frame, textvariable=self.doc_name_manual, width=30).pack(side=tk.LEFT, padx=5)
        ttk.Label(name_frame, text="Промпт:").pack(side=tk.LEFT)
        self.title_prompt_entry = ttk.Entry(name_frame, textvariable=self.title_prompt, width=40)
        self.title_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Сохранение
        save_frame = ttk.Frame(self)
        save_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Checkbutton(save_frame, text="Сохранять PNG", variable=self.save_images_files).pack(side=tk.LEFT)
        ttk.Checkbutton(save_frame, text="Base64", variable=self.embed_images).pack(side=tk.LEFT, padx=10)

        # Управление и прогресс
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        self.btn_build = ttk.Button(ctrl_frame, text="Создать HTML", command=self.start)
        self.btn_build.pack(side=tk.LEFT, padx=2)
        self.btn_pause = ttk.Button(ctrl_frame, text="Пауза", command=self.pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=2)
        self.btn_resume = ttk.Button(ctrl_frame, text="Продолжить", command=self.resume, state=tk.DISABLED)
        self.btn_resume.pack(side=tk.LEFT, padx=2)
        self.btn_stop = ttk.Button(ctrl_frame, text="Стоп", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=2)
        self.progress = ttk.Progressbar(ctrl_frame, length=250, mode='determinate')
        self.progress.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.status_label = ttk.Label(ctrl_frame, text="Готов")
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Лог
        log_frame = ttk.LabelFrame(self, text="Лог", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)

        self.toggle_source()
        self.toggle_mode()
        self.toggle_pages()
        self.toggle_nested()

        # Инициализация промптов
        self.prompt_text.insert('1.0', self.last_prompt_full)
        self._prev_prompt_mode = 'full'

        # Улучшенная поддержка буфера обмена
        self.add_clipboard_support(self.prompt_text)
        self.add_clipboard_support(self.title_prompt_entry)

    def add_clipboard_support(self, widget):
        """Добавляет горячие клавиши Ctrl+C/V/X/A и контекстное меню для любого виджета ввода."""
        def show_menu(event):
            menu = tk.Menu(widget, tearoff=0)
            menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
            menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
            menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
            menu.add_separator()
            menu.add_command(label="Выделить всё", command=lambda: widget.event_generate("<<SelectAll>>"))
            menu.tk_popup(event.x_root, event.y_root)

        widget.bind("<Button-3>", show_menu)
        widget.bind("<Control-c>", lambda e: widget.event_generate("<<Copy>>"))
        widget.bind("<Control-v>", lambda e: widget.event_generate("<<Paste>>"))
        widget.bind("<Control-x>", lambda e: widget.event_generate("<<Cut>>"))
        widget.bind("<Control-a>", lambda e: widget.event_generate("<<SelectAll>>"))

    def _on_prompt_mode_changed(self, *args):
        current_text = self.prompt_text.get('1.0', 'end-1c')
        prev_mode = getattr(self, '_prev_prompt_mode', 'full')
        if prev_mode == 'full':
            self.last_prompt_full = current_text
        else:
            self.last_prompt_ocr = current_text

        new_mode = self.processor_mode.get()
        if new_mode == 'full':
            new_text = self.last_prompt_full
        else:
            new_text = self.last_prompt_ocr

        self.prompt_text.delete('1.0', tk.END)
        self.prompt_text.insert('1.0', new_text)
        self._prev_prompt_mode = new_mode

    # UI helpers
    def toggle_source(self):
        pass

    def toggle_pages(self):
        if self.find_pages.get():
            self.pages_frame.grid()
        else:
            self.pages_frame.grid_remove()

    def toggle_nested(self):
        if self.find_nested.get():
            self.nested_frame.grid()
        else:
            self.nested_frame.grid_remove()

    def browse(self):
        if self.source_mode.get() == "file":
            path = filedialog.askopenfilename(filetypes=[("PNG", "*.png")])
            if path:
                self.input_path.set(path)
        else:
            path = filedialog.askdirectory()
            if path:
                self.input_path.set(path)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("PNG", "*.png")])
        if files:
            self.selected_files = list(files)
            if files:
                self.input_path.set(os.path.dirname(files[0]) + " (выбрано {})".format(len(files)))
                self.source_mode.set("folder")

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
        self.log_func(msg)

    def toggle_mode(self):
        if self.recognition_mode.get() == "lm":
            self.lm_frame.pack(fill=tk.X, padx=10, pady=2, after=self.lm_frame.master.winfo_children()[-1])
        else:
            self.lm_frame.pack_forget()

    # Управление потоком
    def start(self):
        path = self.input_path.get().strip()
        if not path or not os.path.exists(path):
            if not self.selected_files:
                messagebox.showerror("Ошибка", "Неверный путь.")
                return
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена.")
            return

        # Сохраняем текущий промпт в соответствующий last_prompt_*
        current_text = self.prompt_text.get('1.0', 'end-1c')
        mode = self.processor_mode.get()
        if mode == 'full':
            self.last_prompt_full = current_text
        else:
            self.last_prompt_ocr = current_text
        self.prompt_var.set(current_text)

        self.paused.set()
        self.stopped.clear()
        self.btn_build.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.progress['value'] = 0
        self.status_label.config(text="Запуск...")
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def pause(self):
        self.paused.clear()
        self.btn_pause.config(state=tk.DISABLED)
        self.btn_resume.config(state=tk.NORMAL)
        self.log("⏸ Пауза")

    def resume(self):
        self.paused.set()
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_resume.config(state=tk.DISABLED)
        self.log("▶ Продолжение")

    def stop(self):
        self.stopped.set()
        self.paused.set()
        self.log("⏹ Останов")

    def check_state(self):
        if self.stopped.is_set():
            return True
        self.paused.wait()
        return self.stopped.is_set()

    # ---------- ПОСТРОЕНИЕ HTML СТРАНИЦЫ ----------
    def _build_page_html(self, page_idx, nested_infos, raw, success):
        """Формирует HTML-блок страницы на основе raw-ответа и вложенных изображений."""
        if not success or not raw or not raw.strip():
            return f'<div class="page" id="page{page_idx}"><h2>Страница {page_idx}</h2><p>Нет ответа от LM Studio</p></div>', False

        raw = re.sub(r'```html|```', '', raw, flags=re.MULTILINE).strip()

        if nested_infos:
            used = set()
            for m in re.finditer(r'<img[^>]+src="([^"]+)"', raw):
                used.add(os.path.basename(m.group(1)))
            missing = [info for info in nested_infos if info['file'] not in used]
            if missing:
                self.log(f"  ⚠ LM пропустила изображения: {[m['file'] for m in missing]}, добавляю автоматически")
                for info in missing:
                    if self.embed_images.get() and info['b64']:
                        raw += f'<img src="data:image/png;base64,{info["b64"]}" alt="{info["file"]}"><br>\n'
                    elif self.save_images_files.get():
                        raw += f'<img src="{info["file"]}" alt="{info["file"]}"><br>\n'

        # Замена имён файлов на base64, если включено
        if self.embed_images.get():
            for info in nested_infos:
                if info['b64']:
                    raw = raw.replace(info['file'], f"data:image/png;base64,{info['b64']}")

        if not re.search(r'<\w+[^>]*>', raw):
            raw = '<p>' + raw.replace('\n', '</p>\n<p>') + '</p>'

        html = f'<div class="page" id="page{page_idx}">\n<h2>Страница {page_idx}</h2>\n{raw}\n</div>'
        return html, True

    def _replace_page_in_html(self, html_path, page_idx, new_page_html):
        """Читает HTML-файл, заменяет блок страницы и сохраняет обратно."""
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = re.compile(
            r'<div class="page" id="page{}">.*?</div>\s*(?=\n|$)'.format(page_idx),
            re.DOTALL
        )
        if not pattern.search(content):
            # fallback: по заголовку
            pattern = re.compile(
                r'<div class="page">\s*<h2>Страница {}</h2>.*?</div>\s*(?=\n|$)'.format(page_idx),
                re.DOTALL
            )
        new_content = pattern.sub(new_page_html, content, count=1)
        if new_content == content:
            self.log(f"  ⚠ Не удалось найти блок страницы {page_idx} в {html_path}")
            return
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    def _retry_failed_pages(self, all_failed):
        """Фоновый поток: повторяет запросы к LM Studio и обновляет HTML-файлы."""
        self.log("🚀 Начинаем повторную отправку неудачных страниц...")
        for i, fail in enumerate(all_failed, 1):
            self.log(f"Повторная попытка для страницы {fail['page_idx']} (файл {os.path.basename(fail['file_path'])})")
            base_prompt = fail['prompt']
            img_section = ""
            if fail['nested_infos']:
                if self.processor_mode.get() == "full":
                    img_section = f"\n\nЯ нашёл на странице {len(fail['nested_infos'])} изображений. По порядку сверху вниз:\n"
                    for j, info in enumerate(fail['nested_infos'], 1):
                        img_section += f"{j}: \"{info['file']}\"\n"
                    img_section += "Вставь их в HTML с помощью <img src=\"имя_файла\">. Используй ТОЛЬКО указанные имена файлов."
                else:
                    img_section = "\n\nИзображения на странице:\n"
                    for j, info in enumerate(fail['nested_infos'], 1):
                        img_section += f"{j}: \"{info['file']}\"\n"
            final_prompt = base_prompt + img_section

            raw, elapsed = ask_lm_studio(
                fail['crop'], final_prompt, DEFAULT_LM_STUDIO_URL, DEFAULT_LM_MODEL,
                max_tokens=self.max_tokens_var.get(), timeout=self.timeout_var.get()
            )
            self.log(f"  ⏱ Повторный ответ за {elapsed:.1f} сек")

            success = raw is not None and raw.strip() != ""
            new_html, success = self._build_page_html(
                fail['page_idx'], fail['nested_infos'], raw, success
            )
            if success:
                html_path = os.path.join(fail['out_dir'], f"{fail['doc_title']}.html")
                if os.path.exists(html_path):
                    self._replace_page_in_html(html_path, fail['page_idx'], new_html)
                    self.log(f"  ✅ Страница {fail['page_idx']} обновлена в {html_path}")
                else:
                    self.log(f"  ❌ HTML-файл не найден: {html_path}")
            else:
                self.log(f"  ❌ Повторная попытка для страницы {fail['page_idx']} не удалась")
        self.log("✅ Повторная обработка завершена")

    def _show_retry_dialog(self, all_failed):
        """Вызывается в главном потоке после завершения всей обработки."""
        if not all_failed:
            return
        answer = messagebox.askyesno(
            "Повторить неудачные страницы",
            f"Обнаружено {len(all_failed)} страниц без распознанного текста.\n"
            "Хотите отправить их в LM Studio повторно с текущими настройками?"
        )
        if answer:
            threading.Thread(target=self._retry_failed_pages, args=(all_failed,), daemon=True).start()

    # ---------- ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ СТРАНИЦЫ ----------
    def process_page(self, page_img, page_idx, out_dir, nested_infos):
        mode = self.recognition_mode.get()
        if mode == "none":
            imgs = ""
            for info in nested_infos:
                if self.embed_images.get() and info['b64']:
                    imgs += f'<img src="data:image/png;base64,{info["b64"]}" alt="{info["file"]}"><br>\n'
                elif self.save_images_files.get():
                    imgs += f'<img src="{info["file"]}" alt="{info["file"]}"><br>\n'
            html = f'<div class="page" id="page{page_idx}">\n<h2>Страница {page_idx}</h2>\n{imgs}\n</div>'
            self._save_page_html(page_idx, out_dir, html)
            return html, True

        # LM Studio
        base_prompt = self.prompt_var.get()
        img_section = ""
        if nested_infos:
            if self.processor_mode.get() == "full":
                img_section = f"\n\nЯ нашёл на странице {len(nested_infos)} изображений. По порядку сверху вниз:\n"
                for i, info in enumerate(nested_infos, 1):
                    img_section += f"{i}: \"{info['file']}\"\n"
                img_section += "Вставь их в HTML с помощью <img src=\"имя_файла\">. Используй ТОЛЬКО указанные имена файлов."
            else:  # ocr
                img_section = "\n\nИзображения на странице:\n"
                for i, info in enumerate(nested_infos, 1):
                    img_section += f"{i}: \"{info['file']}\"\n"
        final_prompt = base_prompt + img_section

        self.status_label.config(text=f"Страница {page_idx}: запрос к LM...")
        self.log(f"  📡 Запрос LM Studio (токенов: {self.max_tokens_var.get()}, таймаут: {self.timeout_var.get()}с)")
        self.log(f"  Промпт: {final_prompt[:250]}...")

        raw, elapsed = ask_lm_studio(page_img, final_prompt, DEFAULT_LM_STUDIO_URL, DEFAULT_LM_MODEL,
                                     max_tokens=self.max_tokens_var.get(), timeout=self.timeout_var.get())
        self.log(f"  ⏱ Ответ получен за {elapsed:.1f} сек")

        success = raw is not None and raw.strip() != ""
        if raw is None:
            self.log("  ❌ Пустой ответ")
        else:
            self.log(f"  📝 Ответ ({len(raw)} символов): {raw[:200]}...")

        html, success = self._build_page_html(page_idx, nested_infos, raw, success)
        self._save_page_html(page_idx, out_dir, html)
        return html, success

    def _save_page_html(self, page_idx, out_dir, page_html):
        """Сохраняет HTML отдельной страницы в out_dir."""
        path = os.path.join(out_dir, f"page_{page_idx}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(page_html)

    def process_one_file(self, file_path, file_idx, total, start_time):
        """Обработка одного файла. Возвращает (html_parts, failed_pages_info, out_dir, doc_title)."""
        self.status_label.config(text=f"Файл {file_idx}/{total}: {os.path.basename(file_path)}")
        self.log(f"--- Файл {file_idx}/{total}: {os.path.basename(file_path)} ---")
        img = Image.open(file_path).convert("RGB")

        if self.find_pages.get():
            self.status_label.config(text="Поиск страниц...")
            pages = predict_boxes(self.model, img, self.device, self.confidence_pages.get())
            if not pages:
                pages = [(0, 0, img.width, img.height, 1.0)]
            self.log(f"  Найдено страниц: {len(pages)}")
        else:
            pages = [(0, 0, img.width, img.height, 1.0)]

        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        tmp_dir = os.path.join(self.output_dir.get(), clean_filename(base_filename) or "temp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Название
        doc_title = self.doc_name_manual.get().strip()
        if not doc_title and self.recognition_mode.get() == "lm":
            self.status_label.config(text="Запрос названия...")
            try:
                first_crop = img.crop(pages[0][:4])
                title_raw, _ = ask_lm_studio(first_crop, self.title_prompt.get(),
                    DEFAULT_LM_STUDIO_URL, DEFAULT_LM_MODEL,
                    max_tokens=self.max_tokens_var.get(), timeout=self.timeout_var.get())
                if title_raw:
                    doc_title = clean_filename(title_raw)
                    self.log(f"  Название: {doc_title}")
            except:
                pass
        if not doc_title:
            doc_title = base_filename

        out_dir = os.path.join(self.output_dir.get(), doc_title)
        if tmp_dir != out_dir:
            if os.path.exists(out_dir):
                counter = 1
                while os.path.exists(out_dir + f"_{counter}"):
                    counter += 1
                out_dir += f"_{counter}"
            os.rename(tmp_dir, out_dir)
        else:
            out_dir = tmp_dir

        html_parts = []
        failed_pages_info = []
        total_pages = len(pages)
        for page_idx, box in enumerate(pages, 1):
            if self.check_state():
                break
            self.status_label.config(text=f"Страница {page_idx}/{total_pages}")
            crop = img.crop(box[:4])
            page_file = f"page_{page_idx}.png"
            crop.save(os.path.join(out_dir, page_file))

            nested_infos = []
            if self.find_nested.get():
                nested = predict_boxes(self.model, crop, self.device, self.confidence_nested.get())
                nested = [b for b in nested if (b[2]-b[0])>15 and (b[3]-b[1])>15]
                nested.sort(key=lambda b: b[1])
                for ni, nb in enumerate(nested, 1):
                    sub_img = crop.crop(nb[:4])
                    fname = f"page_{page_idx}_img{ni}.png"
                    fpath = os.path.join(out_dir, fname)
                    if self.save_images_files.get():
                        sub_img.save(fpath)
                    b64 = None
                    if self.embed_images.get():
                        buf = BytesIO()
                        sub_img.save(buf, format='PNG')
                        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    nested_infos.append({
                        'file': fname, 'path': fpath, 'b64': b64,
                        'bbox': nb[:4], 'size': (nb[2]-nb[0], nb[3]-nb[1])
                    })

            page_html, success = self.process_page(crop, page_idx, out_dir, nested_infos)
            html_parts.append(page_html)
            if not success:
                failed_pages_info.append({
                    'page_idx': page_idx,
                    'crop': crop.copy(),
                    'nested_infos': nested_infos,
                    'prompt': self.prompt_var.get(),
                    'current_html': page_html
                })

        self._save_html(doc_title, out_dir, html_parts)
        return html_parts, failed_pages_info, out_dir, doc_title

    def run(self):
        start_time = time.time()
        try:
            self.log("=== Начало обработки ===")
            files = []
            if self.selected_files:
                files = self.selected_files
            elif self.source_mode.get() == "folder" or os.path.isdir(self.input_path.get()):
                files = [os.path.join(self.input_path.get(), f) for f in os.listdir(self.input_path.get()) if f.lower().endswith('.png')]
                if not files:
                    self.log("❌ Нет PNG-файлов.")
                    return
            else:
                files = [self.input_path.get()]

            total_files = len(files)
            all_failed = []
            for file_idx, file_path in enumerate(files, 1):
                if self.check_state():
                    break
                file_start = time.time()
                _, failed_for_file, out_dir, doc_title = self.process_one_file(file_path, file_idx, total_files, start_time)
                for fail in failed_for_file:
                    fail['file_path'] = file_path
                    fail['out_dir'] = out_dir
                    fail['doc_title'] = doc_title
                    all_failed.append(fail)
                elapsed = time.time() - file_start
                self.log(f"  ⏱ Файл обработан за {elapsed:.1f} сек")
                self.progress['value'] = (file_idx / total_files) * 100

            self.log(f"=== Готово. Общее время: {time.time()-start_time:.1f} сек ===")
            self.status_label.config(text="Готово")
            self.progress['value'] = 100

            if all_failed:
                self.root.after(0, self._show_retry_dialog, all_failed)

        except Exception as e:
            self.log(f"💥 Ошибка: {e}")
        finally:
            self.btn_build.config(state=tk.NORMAL)
            self.btn_pause.config(state=tk.DISABLED)
            self.btn_resume.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)

    def _save_html(self, doc_title, out_dir, parts):
        if not parts:
            return
        html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><title>{doc_title}</title>
<style>body{{font-family:Arial,sans-serif;margin:20px}}.page{{margin-bottom:40px;border:1px solid #ccc;padding:10px}}img{{max-width:100%}}</style>
</head><body><h1>{doc_title}</h1>{"".join(parts)}</body></html>"""
        path = os.path.join(out_dir, f"{doc_title}.html")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        self.log(f"  ✅ HTML сохранён: {path}")

    def set_model(self, model, device):
        self.model = model
        self.device = device


class App:
    def __init__(self, root):
        self.root = root
        root.title("Image Cutter & HTML Builder")
        root.geometry("1000x850")
        frame = ttk.Frame(root)
        frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frame, text="Устройство:").pack(side=tk.LEFT)
        self.dev_var = ttk.Combobox(frame, values=["Auto", "CPU", "CUDA"], state="readonly")
        self.dev_var.current(0)
        self.dev_var.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Применить", command=self.change_device).pack(side=tk.LEFT, padx=5)
        self.lbl = ttk.Label(frame, text="")
        self.lbl.pack(side=tk.LEFT, padx=20)
        self.device = get_safe_device()
        self.model = self.load_model()
        self.tab = ReconstructionTab(root, self.log, root)
        self.tab.pack(fill=tk.BOTH, expand=True)
        self.tab.set_model(self.model, self.device)

    def log(self, msg):
        print(f"[LOG] {msg}")

    def change_device(self):
        c = self.dev_var.get()
        dev = {'CPU': get_safe_device(force_cpu=True),
               'CUDA': torch.device('cuda') if torch.cuda.is_available() else None,
               'Auto': get_safe_device()}.get(c)
        if dev is None:
            messagebox.showwarning("Нет CUDA", "Видеокарта недоступна.")
            return
        if dev != self.device:
            self.device = dev
            self.model = self.load_model()
            self.tab.set_model(self.model, self.device)
            self.lbl.config(text=f"Устройство: {dev}")

    def load_model(self):
        for name in ["best_model.pth", "trained_cutter_model.pth"]:
            path = os.path.join(MODELS_DIR, name)
            if os.path.exists(path):
                try:
                    model = load_model(path, self.device)
                    self.lbl.config(text=f"Модель: {name} | {self.device}")
                    return model
                except:
                    pass
        messagebox.showwarning("Модель не найдена", "best_model.pth или trained_cutter_model.pth не найдены.")
        return None


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()