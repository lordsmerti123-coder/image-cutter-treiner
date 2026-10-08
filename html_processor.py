import os
import re
import threading
import webbrowser
import shutil
import base64
from datetime import datetime
from tkinter import *
from tkinter import ttk, filedialog, messagebox, scrolledtext
from urllib.parse import urlparse, unquote
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------- Оптимизатор изображений ----------------------------
class ImageOptimizer:
    @staticmethod
    def extract_and_save_images(html_content, base_dir, html_filename):
        """
        Извлекает все изображения из HTML:
        - data:image/png;base64,... -> сохраняет как файл, заменяет на локальный путь
        - внешние url (http://, https://) -> скачивает и сохраняет локально, заменяет
        - уже локальные пути (относительные) оставляет как есть (предполагая, что они уже существуют)
        Возвращает (новый_html, список сохранённых файлов)
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        img_tags = soup.find_all('img')
        saved_images = []
        base_path = Path(base_dir)
        # Создаём папку для изображений рядом с html, например "page_3_files"
        img_folder_name = Path(html_filename).stem + "_files"
        img_dir = base_path / img_folder_name
        img_dir.mkdir(exist_ok=True)

        for idx, img in enumerate(img_tags):
            src = img.get('src', '')
            if not src:
                continue

            # Пропускаем, если src уже указывает на локальный файл внутри папки _files (не нужно повторно извлекать)
            if src.startswith(img_folder_name + '/'):
                continue

            # Обработка base64
            if src.startswith('data:image'):
                # Извлекаем расширение и данные
                match = re.match(r'data:image/(?P<ext>\w+);base64,(?P<data>.+)', src)
                if match:
                    ext = match.group('ext')
                    img_data = base64.b64decode(match.group('data'))
                    new_filename = f"image_{idx}.{ext}"
                    new_path = img_dir / new_filename
                    with open(new_path, 'wb') as f:
                        f.write(img_data)
                    new_src = f"{img_folder_name}/{new_filename}"
                    img['src'] = new_src
                    saved_images.append(str(new_path))
                    continue

            # Обработка внешних URL
            if src.startswith(('http://', 'https://')):
                try:
                    response = requests.get(src, timeout=10)
                    response.raise_for_status()
                    # Определяем расширение из Content-Type или из URL
                    content_type = response.headers.get('content-type', '')
                    ext = 'png'
                    if 'jpeg' in content_type or 'jpg' in content_type:
                        ext = 'jpg'
                    elif 'png' in content_type:
                        ext = 'png'
                    elif 'gif' in content_type:
                        ext = 'gif'
                    else:
                        # По расширению в URL
                        url_path = urlparse(src).path
                        if '.' in url_path:
                            ext = url_path.split('.')[-1].lower()
                            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                                ext = 'png'
                        else:
                            ext = 'png'
                    new_filename = f"image_{idx}.{ext}"
                    new_path = img_dir / new_filename
                    with open(new_path, 'wb') as f:
                        f.write(response.content)
                    new_src = f"{img_folder_name}/{new_filename}"
                    img['src'] = new_src
                    saved_images.append(str(new_path))
                except Exception as e:
                    print(f"Не удалось скачать {src}: {e}")
                    # Оставляем оригинальный src
                    continue

            # Локальные пути (относительные) оставляем как есть, но гарантируем, что они существуют? Проверять не будем.

        return str(soup), saved_images

    @staticmethod
    def copy_images_for_merge(output_dir, processed_files):
        """
        Копирует все изображения из папок _files каждой обработанной страницы в единую папку
        в целевой директории с объединённым HTML и возвращает новый HTML с обновлёнными путями.
        """
        # output_dir - папка, куда сохраняется combined.html
        # создаём внутри output_dir папку combined_files
        combined_img_dir = Path(output_dir) / "combined_files"
        combined_img_dir.mkdir(exist_ok=True)

        merged_html_parts = []
        for filepath in processed_files:
            # Читаем HTML
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            # Для каждого img проверяем src, если он ведёт на локальную папку _files
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src:
                    continue
                # Определяем оригинальный файл изображения
                # Путь может быть относительно обрабатываемого HTML
                html_dir = Path(filepath).parent
                potential_img_path = html_dir / src
                if potential_img_path.exists() and potential_img_path.is_file():
                    # Копируем в combined_img_dir с сохранением имени
                    img_filename = potential_img_path.name
                    dest_path = combined_img_dir / img_filename
                    # Если такой файл уже есть, возможно перезапись (имена могут совпадать, тогда берём последний)
                    shutil.copy2(potential_img_path, dest_path)
                    # Обновляем src в HTML на путь относительно объединённого файла
                    img['src'] = f"combined_files/{img_filename}"
                # Если src - внешняя ссылка, оставляем как есть

            merged_html_parts.append(str(soup))

        # Собираем общий HTML (как в предыдущей версии)
        combined = """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Объединённые страницы</title>
<style>
.page-section { margin-bottom: 40px; border-bottom: 2px solid #ccc; padding-bottom: 20px; }
.page-title { font-size: 1.8em; background-color: #f0f0f0; padding: 8px; margin: 0 0 15px 0; }
</style>
</head>
<body>
"""
        for i, body_html in enumerate(merged_html_parts):
            name = os.path.basename(file_list[i])
            combined += f'<div class="page-section"><div class="page-title">Страница {i+1}: {name}</div>{body_html}</div>\n'
        combined += "</body></html>"
        return combined

# ---------------------------- Клиент LM Studio (тот же) ----------------------------
class LMStudioClient:
    def __init__(self, base_url="http://127.0.0.1", port=1234, model="local-model",
                 max_tokens=10000, temperature=0.7, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.port = port
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.api_url = f"{self.base_url}:{self.port}/v1/chat/completions"

    def send_request(self, system_prompt, user_prompt, html_content):
        full_user_prompt = f"{user_prompt}\n\nHTML для обработки:\n```html\n{html_content}\n```"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
            return reply.strip()
        except Exception as e:
            print(f"Ошибка запроса к LM Studio: {e}")
            if hasattr(e, 'response') and e.response and e.response.text:
                print("Ответ сервера:", e.response.text[:500])
            return None

    def test_connection(self):
        try:
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
                "temperature": 0.0
            }
            resp = requests.post(self.api_url, json=test_payload, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

# ---------------------------- Основное приложение с интеграцией ImageOptimizer ----------------------------
class HTMLProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HTML Processor with LM Studio - с оптимизацией изображений")
        self.root.geometry("950x850")

        self.input_files = []
        self.processed_files = []

        # Настройки
        self.url_var = StringVar(value="http://127.0.0.1")
        self.port_var = StringVar(value="1234")
        self.model_var = StringVar(value="local-model")
        self.max_tokens_var = IntVar(value=10000)
        self.temperature_var = DoubleVar(value=0.7)
        self.timeout_var = IntVar(value=300)

        default_prompt = """Ты редактор документов. На этом изображении — одна страница документа. Далее идёт текст из старого отсканированного и распознанного документа.
Нужно актуализировать текст: исправить ошибки распознавания, дополнить недостающее, сделать удобную читаемую страницу. Не теряй текст в середине. Переосмысли вёрстку. Если в оригинале есть изображения — перенеси их. На выходе должен быть HTML.
Создай html-страницу с содержимым исходного документа. Примени современный дизайн, сделай информацию легко читаемой. Удали внешние ссылки. Изображения оставь на нужных местах и подпиши их по смыслу."""

        self.prompt_text = StringVar(value=default_prompt)

        self.build_ui()

    def build_ui(self):
        # (интерфейс полностью идентичен предыдущему, без изменений, но для краткости приведу только ключевые элементы)
        settings_frame = LabelFrame(self.root, text="Настройки LM Studio", padx=5, pady=5)
        settings_frame.pack(fill="x", padx=10, pady=5)

        row0 = Frame(settings_frame)
        row0.pack(fill="x", pady=2)
        Label(row0, text="URL:").pack(side="left")
        Entry(row0, textvariable=self.url_var, width=20).pack(side="left", padx=5)
        Label(row0, text="Порт:").pack(side="left")
        Entry(row0, textvariable=self.port_var, width=6).pack(side="left", padx=5)
        Label(row0, text="Модель:").pack(side="left")
        Entry(row0, textvariable=self.model_var, width=20).pack(side="left", padx=5)
        Button(row0, text="Проверить соединение", command=self.test_connection).pack(side="left", padx=10)

        row1 = Frame(settings_frame)
        row1.pack(fill="x", pady=2)
        Label(row1, text="Max tokens:").pack(side="left")
        Entry(row1, textvariable=self.max_tokens_var, width=8).pack(side="left", padx=5)
        Label(row1, text="Temperature:").pack(side="left")
        Entry(row1, textvariable=self.temperature_var, width=8).pack(side="left", padx=5)
        Label(row1, text="Timeout (сек):").pack(side="left")
        Entry(row1, textvariable=self.timeout_var, width=8).pack(side="left", padx=5)

        # Панель исходных файлов
        files_frame = LabelFrame(self.root, text="Исходные HTML файлы", padx=5, pady=5)
        files_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.files_listbox = Listbox(files_frame, selectmode=EXTENDED, height=8)
        self.files_listbox.pack(side="left", fill="both", expand=True)
        scroll_files = Scrollbar(files_frame, orient="vertical", command=self.files_listbox.yview)
        scroll_files.pack(side="right", fill="y")
        self.files_listbox.config(yscrollcommand=scroll_files.set)
        btn_frame = Frame(files_frame)
        btn_frame.pack(side="bottom", fill="x", pady=5)
        Button(btn_frame, text="Добавить файлы", command=self.add_files).pack(side="left", padx=5)
        Button(btn_frame, text="Удалить выбранные", command=self.remove_selected_files).pack(side="left", padx=5)

        # Промпт
        prompt_frame = LabelFrame(self.root, text="Промпт для обработки", padx=5, pady=5)
        prompt_frame.pack(fill="x", padx=10, pady=5)
        self.prompt_entry = scrolledtext.ScrolledText(prompt_frame, height=8, wrap=WORD)
        self.prompt_entry.pack(fill="x", padx=5, pady=5)
        self.prompt_entry.insert("1.0", self.prompt_text.get())

        # Кнопка обработки и прогресс
        process_frame = Frame(self.root)
        process_frame.pack(fill="x", padx=10, pady=5)
        self.process_btn = Button(process_frame, text="Начать обработку", command=self.start_processing)
        self.process_btn.pack(side="left", padx=5)
        self.progress_bar = ttk.Progressbar(process_frame, mode="determinate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)
        self.status_label = Label(process_frame, text="Готов")
        self.status_label.pack(side="right", padx=5)

        # Панель результатов
        result_frame = LabelFrame(self.root, text="Обработанные файлы (порядок для сшивки)", padx=5, pady=5)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.result_listbox = Listbox(result_frame, selectmode=EXTENDED, height=8)
        self.result_listbox.pack(side="left", fill="both", expand=True)
        scroll_res = Scrollbar(result_frame, orient="vertical", command=self.result_listbox.yview)
        scroll_res.pack(side="right", fill="y")
        self.result_listbox.config(yscrollcommand=scroll_res.set)
        result_btn_frame = Frame(result_frame)
        result_btn_frame.pack(side="bottom", fill="x", pady=5)
        Button(result_btn_frame, text="↑ Вверх", command=self.move_up).pack(side="left", padx=2)
        Button(result_btn_frame, text="↓ Вниз", command=self.move_down).pack(side="left", padx=2)
        Button(result_btn_frame, text="Открыть выбранный в браузере", command=self.open_selected_processed).pack(side="left", padx=5)
        Button(result_btn_frame, text="Удалить выбранные", command=self.remove_selected_processed).pack(side="left", padx=5)
        Button(result_btn_frame, text="Сшить ВСЕ", command=self.combine_all_pages).pack(side="left", padx=5)
        Button(result_btn_frame, text="Сшить ВЫБРАННЫЕ", command=self.combine_selected_pages).pack(side="left", padx=5)

    # ---------------------------- Методы работы с файлами (без изменений) ----------------------------
    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("HTML files", "*.html *.htm")])
        for f in files:
            if f not in self.input_files:
                self.input_files.append(f)
                self.files_listbox.insert(END, os.path.basename(f))
        self.update_status(f"Добавлено файлов: {len(files)}")

    def remove_selected_files(self):
        selected = self.files_listbox.curselection()
        for idx in reversed(selected):
            del self.input_files[idx]
            self.files_listbox.delete(idx)
        self.update_status(f"Удалено файлов: {len(selected)}")

    def remove_selected_processed(self):
        selected = self.result_listbox.curselection()
        if not selected:
            messagebox.showinfo("Нет выбора", "Выберите файлы для удаления из списка сшивки.")
            return
        for idx in reversed(selected):
            del self.processed_files[idx]
        self.update_processed_listbox()
        self.update_status(f"Удалено обработанных файлов из списка: {len(selected)}")

    # ---------------------------- Обработка с оптимизацией ----------------------------
    def start_processing(self):
        if not self.input_files:
            messagebox.showwarning("Нет файлов", "Добавьте хотя бы один HTML файл.")
            return
        prompt = self.prompt_entry.get("1.0", END).strip()
        if not prompt:
            messagebox.showwarning("Нет промпта", "Введите промпт для обработки.")
            return

        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Порт должен быть числом.")
            return

        self.lm_client = LMStudioClient(
            base_url=self.url_var.get(),
            port=port,
            model=self.model_var.get(),
            max_tokens=self.max_tokens_var.get(),
            temperature=self.temperature_var.get(),
            timeout=self.timeout_var.get()
        )

        self.process_btn.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(self.input_files)
        self.status_label.config(text="Обработка...")
        threading.Thread(target=self.process_files, args=(prompt,), daemon=True).start()

    def process_files(self, prompt):
        system_prompt = ("Ты – ассистент, который возвращает только исправленный HTML-код без лишних пояснений. "
                         "Сохраняй все теги <img> с атрибутами src в том виде, как они есть. Не меняй пути к изображениям.")
        processed = []
        failed_files = []

        for idx, filepath in enumerate(self.input_files):
            self.root.after(0, self.update_progress, idx + 1, f"Оптимизация и обработка: {os.path.basename(filepath)}")

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    original_html = f.read()
            except Exception as e:
                failed_files.append((idx, filepath, f"Ошибка чтения: {e}"))
                continue

            # ----- Оптимизация изображений -----
            base_dir = os.path.dirname(filepath)
            html_filename = os.path.basename(filepath)
            optimized_html, saved_images = ImageOptimizer.extract_and_save_images(original_html, base_dir, html_filename)
            if saved_images:
                self.root.after(0, self.update_status, f"Извлечено {len(saved_images)} изображений из {os.path.basename(filepath)}")

            # Отправляем оптимизированный HTML
            success = False
            retry_count = 0
            while not success and retry_count < 3:
                response = self.lm_client.send_request(system_prompt, prompt, optimized_html)
                if response is None:
                    retry_count += 1
                    if retry_count >= 3:
                        if messagebox.askyesno("Ошибка", f"Повторить для {os.path.basename(filepath)}?"):
                            retry_count = 0
                        else:
                            failed_files.append((idx, filepath, "Сетевая ошибка"))
                            success = True
                else:
                    cleaned = self.extract_html(response)
                    if cleaned:
                        # Сохраняем обработанный HTML в ту же папку, где лежат изображения
                        out_path = os.path.join(base_dir, f"{os.path.splitext(html_filename)[0]}_обработано.html")
                        with open(out_path, 'w', encoding='utf-8') as outf:
                            outf.write(cleaned)
                        processed.append(out_path)
                        success = True
                    else:
                        retry_count += 1
                        if retry_count >= 3:
                            if messagebox.askyesno("Невалидный ответ", f"Повторить для {os.path.basename(filepath)}?"):
                                retry_count = 0
                            else:
                                failed_files.append((idx, filepath, "Ответ не содержит HTML"))
                                success = True

        self.processed_files = processed
        self.root.after(0, self.update_processed_listbox)

        if failed_files:
            msg = "Не обработаны:\n" + "\n".join(f"- {os.path.basename(f[1])}: {f[2]}" for f in failed_files)
            self.root.after(0, lambda: messagebox.showwarning("Частичная ошибка", msg))

        self.root.after(0, self.finish_processing)

    def extract_html(self, text):
        match = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if text.strip().startswith('<'):
            return text.strip()
        return None

    def update_progress(self, value, status_msg):
        self.progress_bar["value"] = value
        self.status_label.config(text=status_msg)

    def update_processed_listbox(self):
        self.result_listbox.delete(0, END)
        for path in self.processed_files:
            self.result_listbox.insert(END, os.path.basename(path))

    def finish_processing(self):
        self.process_btn.config(state=NORMAL)
        self.status_label.config(text="Обработка завершена")
        self.progress_bar["value"] = self.progress_bar["maximum"]
        messagebox.showinfo("Готово", f"Обработано: {len(self.processed_files)}")

    # ---------------------------- Порядок и просмотр ----------------------------
    def move_up(self):
        sel = self.result_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        self.processed_files[idx], self.processed_files[idx-1] = self.processed_files[idx-1], self.processed_files[idx]
        self.update_processed_listbox()
        self.result_listbox.selection_set(idx-1)

    def move_down(self):
        sel = self.result_listbox.curselection()
        if not sel or sel[0] == len(self.processed_files)-1:
            return
        idx = sel[0]
        self.processed_files[idx], self.processed_files[idx+1] = self.processed_files[idx+1], self.processed_files[idx]
        self.update_processed_listbox()
        self.result_listbox.selection_set(idx+1)

    def open_selected_processed(self):
        sel = self.result_listbox.curselection()
        if not sel:
            messagebox.showinfo("Нет выбора", "Выберите файл.")
            return
        path = self.processed_files[sel[0]]
        webbrowser.open(f"file://{os.path.abspath(path)}")

    # ---------------------------- Сшивка с копированием изображений ----------------------------
    def combine_all_pages(self):
        if len(self.processed_files) < 2:
            messagebox.showinfo("Недостаточно файлов", "Нужно минимум 2 файла.")
            return
        self._combine(self.processed_files)

    def combine_selected_pages(self):
        selected_indices = self.result_listbox.curselection()
        if len(selected_indices) < 2:
            messagebox.showinfo("Недостаточно выбрано", "Выберите минимум 2 файла (Ctrl+клик или Shift).")
            return
        files_to_combine = [self.processed_files[i] for i in selected_indices]
        self._combine(files_to_combine)

    def _combine(self, file_list):
        save_dir = filedialog.askdirectory(title="Сохранить объединённый HTML")
        if not save_dir:
            return

        merged_html = ImageOptimizer.copy_images_for_merge(save_dir, file_list)
        if merged_html is None:
            messagebox.showerror("Ошибка", "Не удалось объединить страницы.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"combined_{timestamp}.html")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(merged_html)
        webbrowser.open(f"file://{os.path.abspath(out_path)}")
        messagebox.showinfo("Готово", f"Сохранён: {out_path}")

    def test_connection(self):
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Порт – число.")
            return
        client = LMStudioClient(self.url_var.get(), port, self.model_var.get(),
                                max_tokens=5, temperature=0.0, timeout=10)
        if client.test_connection():
            messagebox.showinfo("Успех", "Соединение с LM Studio работает.")
        else:
            messagebox.showerror("Ошибка", "Не удалось подключиться.\nПроверьте сервер LM Studio и настройки.")

    def update_status(self, msg):
        self.status_label.config(text=msg)

if __name__ == "__main__":
    root = Tk()
    app = HTMLProcessorApp(root)
    root.mainloop()