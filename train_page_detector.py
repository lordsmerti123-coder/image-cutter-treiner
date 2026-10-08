"""
Нейросетевое вырезание картинок (длинные скриншоты)
Переработанная версия – ручное управление скроллингом Canvas,
чтобы избежать бага Tkinter на Windows с yview/xview.
"""
import os, json, shutil, tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import threading
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou
from torchvision import transforms as T
from PIL import Image, ImageTk
import cv2

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------
# Каталоги создаются рядом с этим файлом, чтобы программа работала
# из любой папки.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.join(BASE_DIR, "learning_materials")   # изображения и разметка
MODELS_DIR = os.path.join(BASE_DIR, "models")                 # веса и чекпоинты
LOGS_DIR = os.path.join(BASE_DIR, "logs")                     # журналы обучения
TEMP_DIR = os.path.join(BASE_DIR, "temp")                     # временные файлы
ANNOTATIONS_FILE = os.path.join(LEARNING_DIR, "annotations.json")
MAX_IMAGE_SIDE = 1024          # максимальная сторона изображения, подаваемого в сеть

for d in [LEARNING_DIR, MODELS_DIR, LOGS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def cleanup_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_old_checkpoints(keep_best=True, keep_last=True):
    if not os.path.exists(MODELS_DIR): return
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith(".pth"): continue
        if keep_best and fname == "best_model.pth": continue
        if keep_last and fname == "trained_cutter_model.pth": continue
        if "epoch" in fname or "map" in fname or fname.startswith("model_epoch"):
            os.remove(os.path.join(MODELS_DIR, fname))

def get_safe_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.tensor([1.0], device='cuda')
        return torch.device('cuda')
    except:
        return torch.device('cpu')

# ----------------------------------------------------------------------
# Безопасные обёртки для скроллинга в Text/Listbox (необязательно, но надёжно)
# ----------------------------------------------------------------------
def safe_yview(widget):
    """Возвращает функцию-обёртку для widget.yview, исправляющую одиночный числовой аргумент."""
    def wrapper(*args):
        if args and args[0] not in ('moveto', 'scroll'):
            try:
                float(args[0])
                widget.yview('moveto', *args)
                return
            except ValueError:
                pass
        widget.yview(*args)
    return wrapper

# ----------------------------------------------------------------------
# Масштабирование изображений и боксов
# ----------------------------------------------------------------------
def resize_image_and_boxes(pil_image, boxes, max_side=MAX_IMAGE_SIDE):
    w, h = pil_image.size
    scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
        boxes = [[c * scale for c in box] for box in boxes]
    return pil_image, boxes, scale

def get_transform(pil_image, boxes):
    pil_image, boxes, _ = resize_image_and_boxes(pil_image, boxes)
    img_tensor = T.ToTensor()(pil_image)
    return img_tensor, boxes

def preprocess_for_inference(pil_image):
    pil_image, _, scale = resize_image_and_boxes(pil_image, [])
    img_tensor = T.ToTensor()(pil_image)
    return img_tensor, scale

# ----------------------------------------------------------------------
# Загрузка модели
# ----------------------------------------------------------------------
def load_model_for_inference(model_path, device):
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

# ----------------------------------------------------------------------
# Подсчёт mAP (упрощённый)
# ----------------------------------------------------------------------
def calculate_map(predictions, targets, iou_threshold=0.5):
    all_detections, all_ground_truths = [], []
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        gt_boxes = target['boxes'].cpu().numpy()
        dets = [{'box': b, 'score': s} for b, s in zip(pred_boxes, pred_scores)]
        gts = [{'box': b, 'matched': False} for b in gt_boxes]
        all_detections.append(dets)
        all_ground_truths.append(gts)

    all_scores, all_matched = [], []
    for dets, gts in zip(all_detections, all_ground_truths):
        dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)
        for det in dets_sorted:
            all_scores.append(det['score'])
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gts):
                if not gt['matched']:
                    iou = box_iou(torch.tensor([det['box']]), torch.tensor([gt['box']])).item()
                    if iou > best_iou:
                        best_iou, best_idx = iou, i
            if best_iou >= iou_threshold and best_idx != -1:
                all_matched.append(True)
                gts[best_idx]['matched'] = True
            else:
                all_matched.append(False)

    if not all_scores:
        return 0.0
    tp = np.cumsum(all_matched)
    fp = np.cumsum([not m for m in all_matched])
    precisions = tp / (tp + fp + 1e-6)
    recalls = tp / max(1, sum(len(gt) for gt in all_ground_truths))
    ap = 0.0
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i-1]:
            ap += precisions[i] * (recalls[i] - recalls[i-1])
    return ap

# ----------------------------------------------------------------------
# Датасет
# ----------------------------------------------------------------------
class CutterDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                self.annotations = data.get("images", data)
            except:
                self.annotations = {}
        else:
            self.annotations = {}
        for k, v in self.annotations.items():
            if isinstance(v, dict) and "boxes" in v:
                self.annotations[k] = v["boxes"]
        self.images = [f for f in self.annotations.keys()
                       if os.path.exists(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        pil_img = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        boxes_list = self.annotations[img_name]
        pil_img, boxes_list, _ = resize_image_and_boxes(pil_img, boxes_list)
        if len(boxes_list) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.ones((len(boxes_list),), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        img_tensor = T.ToTensor()(pil_img)
        return img_tensor, target

# ======================================================================
# Вкладка разметки (полная переработка скроллинга Canvas)
# ======================================================================
class AnnotationTab:
    def __init__(self, parent, log_func):
        self.parent = parent
        self.log_func = log_func
        self.image_dir = None
        self.images = []
        self.current_idx = 0
        self.bboxes = []                # ручные боксы в исходных координатах
        self.pred_boxes = []            # предсказания (исходные координаты)
        self.annotations = {}
        self.model = None
        self.device = get_safe_device()
        self.pil_image = None
        self.orig_width = self.orig_height = 0
        self.display_img = None
        self.photo_img = None
        self.canvas_img_id = None
        self.zoom_factor = 1.0
        self.display_scale = 1.0
        self.scale = 1.0                # = display_scale * zoom_factor
        self.start_x = self.start_y = None
        self.current_rect_id = None
        self.rect_ids, self.pred_ids, self.text_ids = [], [], []
        self.setup_gui()
        self.load_model_for_annotation()

    # ------------------------------------------------------------------
    # GUI и скроллинг – главное изменение
    # ------------------------------------------------------------------
    def setup_gui(self):
        top = tk.Frame(self.parent)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.btn_load = tk.Button(top, text="Загрузить папку со скриншотами", command=self.load_folder)
        self.btn_load.pack(side=tk.LEFT, padx=2)

        self.btn_save = tk.Button(top, text="Сохранить аннотации", command=self.save_annotations, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=2)

        self.btn_next = tk.Button(top, text="Следующее", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        self.btn_prev = tk.Button(top, text="Предыдущее", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.btn_del = tk.Button(top, text="Удалить последний", command=self.delete_last_rect, state=tk.DISABLED)
        self.btn_del.pack(side=tk.LEFT, padx=2)

        self.btn_clear = tk.Button(top, text="Очистить всё", command=self.clear_all_rects, state=tk.DISABLED)
        self.btn_clear.pack(side=tk.LEFT, padx=2)

        self.btn_predict = tk.Button(top, text="Предсказать нейросетью", command=self.predict_boxes, state=tk.DISABLED)
        self.btn_predict.pack(side=tk.LEFT, padx=2)

        self.btn_accept = tk.Button(top, text="Принять все предсказания", command=self.accept_all_predictions, state=tk.DISABLED)
        self.btn_accept.pack(side=tk.LEFT, padx=2)

        zoom_frame = tk.Frame(top)
        zoom_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(zoom_frame, text="Масштаб:").pack(side=tk.LEFT)
        self.zoom_slider = tk.Scale(zoom_frame, from_=0.5, to=10.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, length=150, command=self.on_zoom)
        self.zoom_slider.set(1.0)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)
        self.zoom_label = tk.Label(zoom_frame, text="1.0×")
        self.zoom_label.pack(side=tk.LEFT)

        self.info_label = tk.Label(top, text="Нет загруженных изображений")
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Основной контейнер с canvas и скроллбарами
        container = tk.Frame(self.parent)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.h_scroll = tk.Scrollbar(container, orient=tk.HORIZONTAL)
        self.v_scroll = tk.Scrollbar(container, orient=tk.VERTICAL)

        # ОБЫЧНЫЙ Canvas, без автоматической связки со скроллбарами
        self.canvas = tk.Canvas(container, bg='gray', cursor="cross")

        # Скроллбары управляются нашими безопасными функциями
        self.h_scroll.config(command=self._on_hscroll)
        self.v_scroll.config(command=self._on_vscroll)

        # Размещение элементов в grid
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Привязка мыши
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)

        # При изменении размера контейнера обновляем скроллбары
        container.bind("<Configure>", self._on_container_configure)

    def _on_vscroll(self, *args):
        """Безопасный обработчик вертикального скроллбара."""
        if args and args[0] not in ('moveto', 'scroll'):
            try:
                float(args[0])
                self.canvas.yview('moveto', *args)
                return
            except ValueError:
                pass
        self.canvas.yview(*args)

    def _on_hscroll(self, *args):
        """Безопасный обработчик горизонтального скроллбара."""
        if args and args[0] not in ('moveto', 'scroll'):
            try:
                float(args[0])
                self.canvas.xview('moveto', *args)
                return
            except ValueError:
                pass
        self.canvas.xview(*args)

    def _update_scrollbars(self):
        """Синхронизируем положение скроллбаров с фактическим видом canvas."""
        # Метод .yview() / .xview() без аргументов возвращает (first, last)
        self.v_scroll.set(*self.canvas.yview())
        self.h_scroll.set(*self.canvas.xview())

    def _on_container_configure(self, event):
        """Контейнер изменил размер – обновляем скроллбары."""
        self._update_scrollbars()

    # ------------------------------------------------------------------
    # Остальные методы разметки без изменений
    # ------------------------------------------------------------------
    def on_zoom(self, value):
        self.zoom_factor = float(value)
        self.zoom_label.config(text=f"{self.zoom_factor:.1f}×")
        self.update_display()
        self._update_scrollbars()

    def on_mousewheel(self, e):
        self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        self._update_scrollbars()

    def on_shift_mousewheel(self, e):
        self.canvas.xview_scroll(int(-1*(e.delta/120)), "units")
        self._update_scrollbars()

    def update_display(self):
        if self.display_img is None:
            return
        h, w = self.display_img.shape[:2]
        new_w, new_h = int(w * self.zoom_factor), int(h * self.zoom_factor)
        if new_w > 0 and new_h > 0:
            zoomed = cv2.resize(self.display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            self.photo_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)))
            if self.canvas_img_id:
                self.canvas.itemconfig(self.canvas_img_id, image=self.photo_img)
            else:
                self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo_img)
            self.canvas.config(scrollregion=(0, 0, new_w, new_h))
            self.scale = self.display_scale * self.zoom_factor

        # Удаляем старые примитивы
        for rid in self.rect_ids + self.pred_ids + self.text_ids:
            self.canvas.delete(rid)
        self.rect_ids, self.pred_ids, self.text_ids = [], [], []

        # Ручные боксы (красный)
        for (x1, y1, x2, y2) in self.bboxes:
            cx1, cy1, cx2, cy2 = x1*self.scale, y1*self.scale, x2*self.scale, y2*self.scale
            self.rect_ids.append(self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline='red', width=2))

        # Предсказанные боксы (зелёный/жёлтый)
        for (x1, y1, x2, y2, score) in self.pred_boxes:
            cx1, cy1, cx2, cy2 = x1*self.scale, y1*self.scale, x2*self.scale, y2*self.scale
            color = 'green' if score >= 0.7 else 'yellow'
            self.pred_ids.append(self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                                              outline=color, width=2, dash=(4, 2)))
            self.text_ids.append(self.canvas.create_text(cx1+5, cy1+5, anchor='nw',
                                                         text=f"{score:.2f}", fill=color,
                                                         font=('Arial', 8)))

    def load_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с PNG-скриншотами")
        if not folder:
            return
        self.image_dir = folder
        self.images = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        if not self.images:
            messagebox.showerror("Ошибка", "В папке нет PNG-файлов")
            return
        self.images.sort()
        self.current_idx = 0
        self.load_annotations()
        self.load_current_image()
        for btn in (self.btn_save, self.btn_next, self.btn_prev, self.btn_del, self.btn_clear, self.btn_accept):
            btn.config(state=tk.NORMAL)
        self.btn_predict.config(state=tk.NORMAL)
        self.info_label.config(text=f"Файлов: {len(self.images)}")
        self.log_func(f"Загружена папка: {folder}, файлов: {len(self.images)}")

    def load_annotations(self):
        if not os.path.exists(ANNOTATIONS_FILE):
            self.annotations = {}
            return
        try:
            with open(ANNOTATIONS_FILE, 'r') as f:
                data = json.load(f)
            self.annotations = data.get("images", data)
            for k, v in self.annotations.items():
                if isinstance(v, dict) and "boxes" in v:
                    self.annotations[k] = v["boxes"]
        except:
            self.annotations = {}

    def load_current_image(self):
        if not self.images:
            return
        self.canvas.delete("all")
        self.canvas_img_id = self.photo_img = None
        self.rect_ids, self.pred_ids, self.text_ids = [], [], []
        fname = self.images[self.current_idx]
        path = os.path.join(self.image_dir, fname)
        self.pil_image = Image.open(path).convert("RGB")
        self.orig_width, self.orig_height = self.pil_image.size

        max_display = 2000
        self.display_scale = max_display / max(self.orig_width, self.orig_height) if max(self.orig_width, self.orig_height) > max_display else 1.0
        disp_w, disp_h = int(self.orig_width*self.display_scale), int(self.orig_height*self.display_scale)
        pil_disp = self.pil_image.resize((disp_w, disp_h), Image.BICUBIC)
        self.display_img = cv2.cvtColor(np.array(pil_disp), cv2.COLOR_RGB2BGR)

        self.bboxes = [list(box) for box in self.annotations.get(fname, [])]
        self.pred_boxes = []
        self.zoom_slider.set(1.0)
        self.zoom_factor = 1.0
        self.zoom_label.config(text="1.0×")
        self.update_display()
        self._update_scrollbars()
        self.info_label.config(text=f"{self.current_idx+1}/{len(self.images)}: {fname}")

    def save_annotations(self):
        if not self.images:
            return
        curr = self.images[self.current_idx]
        src = os.path.join(self.image_dir, curr)
        dst = os.path.join(LEARNING_DIR, curr)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        self.annotations[curr] = [[int(x) for x in box] for box in self.bboxes]
        with open(ANNOTATIONS_FILE, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        self.log_func(f"Сохранено {curr} ({len(self.bboxes)} прямоугольников)")

    def load_model_for_annotation(self):
        mp = os.path.join(MODELS_DIR, "best_model.pth")
        if not os.path.exists(mp):
            mp = os.path.join(MODELS_DIR, "trained_cutter_model.pth")
        if not os.path.exists(mp):
            self.log_func("Модель не найдена. Обучите модель во вкладке 'Обучение'.")
            self.btn_predict.config(state=tk.DISABLED, text="Нет модели")
            return
        try:
            self.model = load_model_for_inference(mp, self.device)
            self.log_func(f"Модель загружена ({self.device})")
            self.btn_predict.config(state=tk.NORMAL, text="Предсказать нейросетью")
        except Exception as e:
            self.log_func(f"Ошибка загрузки модели: {e}")
            self.model = None
            self.btn_predict.config(state=tk.DISABLED, text="Ошибка модели")

    def predict_boxes(self):
        if self.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена")
            return
        if self.pil_image is None:
            messagebox.showwarning("Нет изображения")
            return
        self.log_func("Предсказание...")
        img_tensor, scale_inv = preprocess_for_inference(self.pil_image)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            preds = self.model([img_tensor])
        boxes = preds[0]['boxes'].cpu().numpy()
        scores = preds[0]['scores'].cpu().numpy()
        self.pred_boxes = []
        for b, s in zip(boxes, scores):
            if s >= 0.3:
                orig_box = [int(coord / scale_inv) for coord in b]
                self.pred_boxes.append((*orig_box, float(s)))
        self.log_func(f"Найдено {len(self.pred_boxes)} объектов")
        self.update_display()
        self._update_scrollbars()

    def accept_all_predictions(self):
        if not self.pred_boxes:
            return
        count = len(self.pred_boxes)
        for box in self.pred_boxes:
            self.bboxes.append(box[:4])
        self.pred_boxes.clear()
        self.update_display()
        self.log_func(f"Принято {count} предсказаний")
        self.save_annotations()

    def on_right_click(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        for i, (x1, y1, x2, y2, _) in enumerate(self.pred_boxes):
            cx1, cy1, cx2, cy2 = x1*self.scale, y1*self.scale, x2*self.scale, y2*self.scale
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                del self.pred_boxes[i]
                self.update_display()
                self.log_func("Предсказание отклонено")
                break

    def on_mouse_down(self, e):
        self.start_x, self.start_y = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        self.current_rect_id = None

    def on_mouse_move(self, e):
        if self.start_x is None:
            return
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, cx, cy, outline='blue', width=2, dash=(4, 2))

    def on_mouse_up(self, e):
        if self.start_x is None:
            return
        ex, ey = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
        x1 = int(min(self.start_x, ex) / self.scale)
        y1 = int(min(self.start_y, ey) / self.scale)
        x2 = int(max(self.start_x, ex) / self.scale)
        y2 = int(max(self.start_y, ey) / self.scale)
        if x2 - x1 > 5 and y2 - y1 > 5:
            self.bboxes.append((x1, y1, x2, y2))
            self.update_display()
            self.log_func(f"Добавлен прямоугольник ({x1},{y1})-({x2},{y2})")
        self.start_x = None

    def delete_last_rect(self):
        if self.bboxes:
            self.bboxes.pop()
            self.update_display()
            self.log_func("Последний прямоугольник удалён")
        else:
            self.log_func("Нет прямоугольников для удаления")

    def clear_all_rects(self):
        self.bboxes.clear()
        self.pred_boxes.clear()
        self.update_display()
        self.log_func("Все прямоугольники удалены")
        if self.images:
            self.info_label.config(text=f"{self.current_idx+1}/{len(self.images)}: {self.images[self.current_idx]}")

    def next_image(self):
        self.save_annotations()
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_current_image()
        else:
            self.info_label.config(text="Это последнее изображение")

    def prev_image(self):
        self.save_annotations()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()

# ======================================================================
# Вкладка обучения
# ======================================================================
class TrainingTab:
    def __init__(self, parent, log_func):
        self.parent = parent
        self.log_func = log_func
        self.device = get_safe_device()
        self.model = None

        param = tk.LabelFrame(parent, text="Параметры обучения", padx=10, pady=10)
        param.pack(fill=tk.X, padx=10, pady=5)

        dev_info = f"Устройство: {self.device}"
        if self.device.type == 'cuda':
            dev_info += f" ({torch.cuda.get_device_name(0)})"
        tk.Label(param, text=dev_info, fg='blue' if self.device.type == 'cuda' else 'red').grid(
            row=0, column=0, columnspan=3, pady=5)

        tk.Label(param, text="Папка с материалами:").grid(row=1, column=0, sticky='e', padx=5)
        tk.Label(param, text=LEARNING_DIR, fg='green').grid(row=1, column=1, sticky='w')
        tk.Button(param, text="Управление материалами", command=self.manage_materials).grid(row=1, column=2, padx=5)

        tk.Label(param, text="Эпохи:").grid(row=2, column=0, sticky='e', padx=5)
        self.epochs_var = tk.IntVar(value=60)
        tk.Entry(param, textvariable=self.epochs_var, width=10).grid(row=2, column=1, sticky='w')

        tk.Label(param, text="Батч:").grid(row=3, column=0, sticky='e', padx=5)
        self.batch_var = tk.IntVar(value=8)
        tk.Entry(param, textvariable=self.batch_var, width=10).grid(row=3, column=1, sticky='w')

        tk.Label(param, text="lr:").grid(row=4, column=0, sticky='e', padx=5)
        self.lr_var = tk.DoubleVar(value=0.005)
        tk.Entry(param, textvariable=self.lr_var, width=10).grid(row=4, column=1, sticky='w')

        self.btn_start = tk.Button(param, text="Начать обучение", command=self.start_training)
        self.btn_start.grid(row=5, column=0, columnspan=3, pady=10)

        self.progress = ttk.Progressbar(parent, length=400, mode='determinate')
        self.progress.pack(pady=10)

        log_frame = tk.LabelFrame(parent, text="Лог обучения", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Scrollbar(log_frame, command=safe_yview(self.log_text)).pack(side=tk.RIGHT, fill=tk.Y)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.parent.update_idletasks()
        self.log_func(msg)

    def manage_materials(self):
        win = tk.Toplevel(self.parent)
        win.title("Управление материалами")
        win.geometry("600x400")
        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        listbox = tk.Listbox(frame, selectmode=tk.EXTENDED)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Scrollbar(frame, command=safe_yview(listbox)).pack(side=tk.RIGHT, fill=tk.Y)
        for f in sorted(f for f in os.listdir(LEARNING_DIR) if f.endswith('.png')):
            listbox.insert(tk.END, f)

        def delete_selected():
            sel = listbox.curselection()
            if not sel:
                return
            to_delete = [listbox.get(idx) for idx in sel]
            for fname in to_delete:
                os.remove(os.path.join(LEARNING_DIR, fname))
            listbox.delete(0, tk.END)
            for f in sorted(f for f in os.listdir(LEARNING_DIR) if f.endswith('.png')):
                listbox.insert(tk.END, f)
            if os.path.exists(ANNOTATIONS_FILE):
                try:
                    with open(ANNOTATIONS_FILE, 'r') as f:
                        ann = json.load(f)
                    for fname in to_delete:
                        ann.pop(fname, None)
                    with open(ANNOTATIONS_FILE, 'w') as f:
                        json.dump(ann, f, indent=2)
                except:
                    pass
            messagebox.showinfo("Удалено", "Выбранные файлы удалены.")
        tk.Button(win, text="Удалить выбранные", command=delete_selected).pack(pady=5)

    def start_training(self):
        if not os.path.exists(ANNOTATIONS_FILE):
            messagebox.showerror("Ошибка", "Нет аннотаций. Сначала сделайте разметку.")
            return
        threading.Thread(target=self.train, daemon=True).start()

    def train(self):
        self.btn_start.config(state=tk.DISABLED, text="Обучение...")
        self.progress['value'] = 0
        self.log("Начало обучения...")
        cleanup_temp()
        self.log(f"Устройство: {self.device}")
        try:
            dataset = CutterDataset(LEARNING_DIR, ANNOTATIONS_FILE)
            n = len(dataset)
            if n < 2:
                self.log(f"Недостаточно данных (найдено {n})")
                return
            n_train = max(1, int(0.8 * n))
            n_val = n - n_train
            if n_val == 0:
                n_val, n_train = 1, n - 1
            self.log(f"Тренировочных: {n_train}, валидационных: {n_val}")
            train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
            train_loader = DataLoader(train_set, batch_size=self.batch_var.get(), shuffle=True,
                                      collate_fn=lambda x: tuple(zip(*x)))
            val_loader = DataLoader(val_set, batch_size=self.batch_var.get(), shuffle=False,
                                    collate_fn=lambda x: tuple(zip(*x)))

            self.model = fasterrcnn_resnet50_fpn(weights=None)
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

            local_weights = os.path.join(MODELS_DIR, "fasterrcnn_resnet50_fpn_coco.pth")
            if os.path.exists(local_weights):
                self.log("Загрузка локальных весов...")
                state = torch.load(local_weights, map_location=self.device)
                state = {k: v for k, v in state.items() if 'roi_heads.box_predictor' not in k}
                self.model.load_state_dict(state, strict=False)
            else:
                self.log("Загрузка весов из интернета...")
                backbone = fasterrcnn_resnet50_fpn(weights='DEFAULT')
                backbone_state = backbone.state_dict()
                backbone_state = {k: v for k, v in backbone_state.items() if 'roi_heads.box_predictor' not in k}
                self.model.load_state_dict(backbone_state, strict=False)
                torch.save(backbone_state, local_weights)
                self.log(f"Веса сохранены в {local_weights}")

            self.model.to(self.device)
            optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                        lr=self.lr_var.get(), momentum=0.9, weight_decay=0.0005)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            writer = SummaryWriter(os.path.join(LOGS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S")))
            best_map = 0.0

            total_batches = len(train_loader) * self.epochs_var.get()
            processed = 0
            for epoch in range(self.epochs_var.get()):
                self.model.train()
                total_loss = 0.0
                for i, (imgs, targets) in enumerate(train_loader):
                    imgs = [img.to(self.device) for img in imgs]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    loss_dict = self.model(imgs, targets)
                    losses = sum(loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    total_loss += losses.item()
                    if i % 5 == 0:
                        self.log(f"Эпоха {epoch+1}, шаг {i}, loss: {losses.item():.4f}")
                    processed += 1
                    if processed % 50 == 0:
                        self.progress['value'] = (processed / total_batches) * 100
                        self.parent.update_idletasks()
                avg_loss = total_loss / len(train_loader)
                self.log(f"Эпоха {epoch+1} завершена. Средний loss: {avg_loss:.4f}")

                self.model.eval()
                preds_all, targets_all = [], []
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs = [img.to(self.device) for img in imgs]
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                        preds_all.extend(self.model(imgs))
                        targets_all.extend(targets)
                cur_map = calculate_map(preds_all, targets_all)
                self.log(f"Валидационный mAP: {cur_map:.4f}")

                writer.add_scalar('Loss/train', avg_loss, epoch)
                writer.add_scalar('mAP/val', cur_map, epoch)
                scheduler.step()

                if cur_map > best_map:
                    best_map = cur_map
                    torch.save(self.model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
                    self.log(f"Новая лучшая модель (mAP={cur_map:.4f})")

            writer.close()
            torch.save(self.model.state_dict(), os.path.join(MODELS_DIR, "trained_cutter_model.pth"))
            self.log("Обучение завершено!")
            cleanup_old_checkpoints(keep_best=True, keep_last=True)
            messagebox.showinfo("Готово", "Модель обучена и сохранена!")
        except Exception as e:
            self.log(f"Ошибка: {e}")
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.btn_start.config(state=tk.NORMAL, text="Начать обучение")
            self.progress['value'] = 100

# ======================================================================
# Главное окно
# ======================================================================
class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("Нейросетевое вырезание картинок (длинные скриншоты)")
        root.geometry("1200x900")
        device = get_safe_device()
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: GPU не используется.")
        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.annotation_tab = AnnotationTab(tk.Frame(notebook), lambda msg: print(f"[LOG] {msg}"))
        notebook.add(self.annotation_tab.parent, text="1. Разметка (прокрутка/масштаб)")

        self.training_tab = TrainingTab(tk.Frame(notebook), lambda msg: print(f"[LOG] {msg}"))
        notebook.add(self.training_tab.parent, text="2. Обучение")

        self.status_var = tk.StringVar(value="Готово. Правый клик — удалить предсказание.")
        tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X)

if __name__ == "__main__":
    cleanup_temp()
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()