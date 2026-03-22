
"""
PatchCore GUI - GPU-lite full app with horizontal gallery thumbnails
- Click a thumbnail to run detection (like VisionPro).
Requirements:
  pip install pyqt5 torch torchvision numpy pillow opencv-python
Run: python app.py
"""

# app.py
"""
PatchCore GUI - GPU-lite full app with horizontal gallery thumbnails (96x96) + filenames
- Click a thumbnail to run detection (like VisionPro).
- Thumbnails show filename below, thin 1px border: OK (green) / NG (red) / Unknown (gray)
- Horizontal scroll bar always visible, small spacing, dark background.
Requirements:
  pip install pyqt5 torch torchvision numpy pillow opencv-python
Run: python app.py
"""

import sys
import os
import time
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QListWidget, QMessageBox, QProgressBar,
    QCheckBox, QGroupBox, QGridLayout, QListWidgetItem, QSizePolicy,
    QScrollArea, QWidgetItem, QFrame, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QSize

# ----------------------------
# PatchCore backbone (ResNet50 layer2 + layer3)
# ----------------------------
try:
    # torchvision new weights
    from torchvision.models import resnet50, ResNet50_Weights
except Exception:
    # fallback
    from torchvision import models
    def resnet50(weights=None):
        return models.resnet50(pretrained=(weights is not None))

class PatchCoreBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            if pretrained:
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                model = resnet50(weights=None)
        except Exception:
            from torchvision import models
            model = models.resnet50(pretrained=pretrained)

        self.stage1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1
        )
        self.stage2 = model.layer2
        self.stage3 = model.layer3

    def forward(self, x):
        x = self.stage1(x)
        f2 = self.stage2(x)
        f3 = self.stage3(f2)
        return f2, f3

# ----------------------------
# Utilities: image preprocess & QPixmap display
# ----------------------------
from torchvision import transforms

_PREPROCESS = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def pil_to_qpixmap(pil_img):
    pil_img = pil_img.convert("RGBA")
    data = pil_img.tobytes("raw", "RGBA")
    qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

# ----------------------------
# Extract patches + memory building + detect functions (core logic)
# ----------------------------
def extract_patches_pytorch(backbone, pil_img, device="cpu"):
    device_t = torch.device(device)
    backbone = backbone.to(device_t)
    backbone.eval()
    with torch.no_grad():
        x = _PREPROCESS(pil_img).unsqueeze(0).to(device_t)  # [1,3,256,256]
        f2, f3 = backbone(x)
        f3r = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        feats = torch.cat([f2, f3r], dim=1)  # [1, C, h, w]
        B, C, h, w = feats.shape
        patches = feats.view(C, h*w).permute(1,0).contiguous()  # (h*w, C)
        norms = patches.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        patches = patches / norms
        patches_cpu = patches.cpu()
    return patches_cpu, (h, w), pil_img.size

# def build_memory_bank_safe(backbone, image_paths, device="cpu", max_patches=20000, progress_callback=None):
#     if progress_callback:
#         progress_callback(0)
#     per_image_patches = []
#     total_imgs = len(image_paths)
#     for i, p in enumerate(image_paths):
#         pil = Image.open(p).convert("RGB")
#         patches_cpu, (h,w), _ = extract_patches_pytorch(backbone, pil, device=device)
#         per_image_patches.append(patches_cpu)
#         if progress_callback:
#             progress_callback(int(40 * (i+1)/total_imgs))
#     if len(per_image_patches) == 0:
#         raise RuntimeError("No patches extracted")
#     memory_bank = torch.cat(per_image_patches, dim=0)
#     N_total = memory_bank.shape[0]
#     if N_total > max_patches:
#         idxs = torch.randperm(N_total)[:max_patches]
#         memory_bank = memory_bank[idxs].contiguous()
#         N_total = memory_bank.shape[0]
#     train_scores = []
#     memory_bank_cpu = memory_bank.float()
#     for i, patches_cpu in enumerate(per_image_patches):
#         if patches_cpu.shape[0] == 0:
#             train_scores.append(0.0)
#             continue
#         try:
#             dists = torch.cdist(patches_cpu.float(), memory_bank_cpu, p=2.0)
#             dist_min, _ = torch.min(dists, dim=1)
#             s_star = float(torch.max(dist_min).item())
#         except RuntimeError:
#             M = memory_bank_cpu.shape[0]
#             batch = 1024
#             min_vals = []
#             for start in range(0, patches_cpu.shape[0], batch):
#                 end = min(patches_cpu.shape[0], start+batch)
#                 A = patches_cpu[start:end].float()
#                 A_sq = (A * A).sum(dim=1, keepdim=True)
#                 min_chunk = None
#                 chunk_size = 4096
#                 for mstart in range(0, M, chunk_size):
#                     mend = min(M, mstart+chunk_size)
#                     B = memory_bank_cpu[mstart:mend]
#                     B_sq = (B * B).sum(dim=1).unsqueeze(0)
#                     AB = A @ B.t()
#                     d2 = A_sq + B_sq - 2.0*AB
#                     if min_chunk is None:
#                         min_chunk = d2.min(dim=1).values
#                     else:
#                         min_chunk = torch.min(min_chunk, d2.min(dim=1).values)
#                 min_vals.append(torch.sqrt(torch.clamp(min_chunk, min=0.0)))
#             dist_min = torch.cat(min_vals, dim=0)
#             s_star = float(torch.max(dist_min).item())
#         train_scores.append(s_star)
#         if progress_callback:
#             progress_callback(40 + int(50 * (i+1)/len(per_image_patches)))
#     best_threshold = float(np.percentile(train_scores, 99)) if len(train_scores)>0 else 0.0
#     memory_bank_np = memory_bank_cpu.numpy()
#     if progress_callback:
#         progress_callback(100)
#     return memory_bank_np, best_threshold, train_scores
# core set 10%
def build_memory_bank_safe(
    backbone,
    image_paths,
    device="cpu",
    max_patches=20000,
    coreset_ratio=0.1,          # ⭐ NEW
    progress_callback=None
):
    if progress_callback:
        progress_callback(0)

    per_image_patches = []
    total_imgs = len(image_paths)

    # -------------------------------------------------
    # 1. Extract patches per image
    # -------------------------------------------------
    for i, p in enumerate(image_paths):
        pil = Image.open(p).convert("RGB")
        patches_cpu, (h, w), _ = extract_patches_pytorch(backbone, pil, device=device)
        per_image_patches.append(patches_cpu)

        if progress_callback:
            progress_callback(int(40 * (i + 1) / total_imgs))

    if len(per_image_patches) == 0:
        raise RuntimeError("No patches extracted")

    # -------------------------------------------------
    # 2. Build raw memory bank
    # -------------------------------------------------
    memory_bank = torch.cat(per_image_patches, dim=0)  # [N, C]
    N_total = memory_bank.shape[0]

    # -------------------------------------------------
    # 3. Hard cap memory (safety)
    # -------------------------------------------------
    if N_total > max_patches:
        idxs = torch.randperm(N_total)[:max_patches]
        memory_bank = memory_bank[idxs].contiguous()
        N_total = memory_bank.shape[0]

    # -------------------------------------------------
    # 4. ⭐ RANDOM CORE-SET (10%)
    # -------------------------------------------------
    if coreset_ratio is not None and 0 < coreset_ratio < 1.0:
        k = max(1, int(N_total * coreset_ratio))
        idxs = torch.randperm(N_total)[:k]
        memory_bank = memory_bank[idxs].contiguous()
        N_total = memory_bank.shape[0]

    memory_bank_cpu = memory_bank.float()  # [M, C]

    # -------------------------------------------------
    # 5. Compute train scores (s*)
    # -------------------------------------------------
    train_scores = []

    for i, patches_cpu in enumerate(per_image_patches):
        if patches_cpu.shape[0] == 0:
            train_scores.append(0.0)
            continue

        try:
            dists = torch.cdist(patches_cpu.float(), memory_bank_cpu, p=2.0)
            dist_min, _ = torch.min(dists, dim=1)
            s_star = float(torch.max(dist_min).item())

        except RuntimeError:
            # fallback chunked distance
            M = memory_bank_cpu.shape[0]
            batch = 1024
            min_vals = []

            for start in range(0, patches_cpu.shape[0], batch):
                end = min(patches_cpu.shape[0], start + batch)
                A = patches_cpu[start:end].float()
                A_sq = (A * A).sum(dim=1, keepdim=True)

                min_chunk = None
                chunk_size = 4096

                for mstart in range(0, M, chunk_size):
                    mend = min(M, mstart + chunk_size)
                    B = memory_bank_cpu[mstart:mend]
                    B_sq = (B * B).sum(dim=1).unsqueeze(0)
                    AB = A @ B.t()
                    d2 = A_sq + B_sq - 2.0 * AB

                    cur_min = d2.min(dim=1).values
                    min_chunk = cur_min if min_chunk is None else torch.min(min_chunk, cur_min)

                min_vals.append(torch.sqrt(torch.clamp(min_chunk, min=0.0)))

            dist_min = torch.cat(min_vals, dim=0)
            s_star = float(torch.max(dist_min).item())

        train_scores.append(s_star)

        if progress_callback:
            progress_callback(40 + int(50 * (i + 1) / len(per_image_patches)))

    # -------------------------------------------------
    # 6. Threshold
    # -------------------------------------------------
    best_threshold = float(np.percentile(train_scores, 99)) if len(train_scores) > 0 else 0.0

    memory_bank_np = memory_bank_cpu.numpy()

    if progress_callback:
        progress_callback(100)

    return memory_bank_np, best_threshold, train_scores


def detect_image_safe(backbone, memory_bank_np, pil_img, device="cpu", interp_size=(224,224)):
    patches_cpu, (h,w), orig_size = extract_patches_pytorch(backbone, pil_img, device=device)
    if patches_cpu.shape[0] == 0:
        raise RuntimeError("No patches in test image")
    memory_bank = torch.from_numpy(memory_bank_np).cpu().float()
    patches = patches_cpu.float()
    try:
        distances = torch.cdist(patches, memory_bank, p=2.0)
        dist_min, _ = torch.min(distances, dim=1)
    except RuntimeError:
        M = memory_bank.shape[0]
        batchA = 1024
        mins = []
        for start in range(0, patches.shape[0], batchA):
            end = min(patches.shape[0], start+batchA)
            A = patches[start:end]
            chunk_min = None
            chunk_size = 4096
            A_sq = (A*A).sum(dim=1, keepdim=True)
            for mstart in range(0, M, chunk_size):
                mend = min(M, mstart+chunk_size)
                B = memory_bank[mstart:mend]
                B_sq = (B*B).sum(dim=1).unsqueeze(0)
                AB = A @ B.t()
                d2 = A_sq + B_sq - 2.0*AB
                cur_min = torch.sqrt(torch.clamp(d2.min(dim=1).values, min=0.0))
                if chunk_min is None:
                    chunk_min = cur_min
                else:
                    chunk_min = torch.min(chunk_min, cur_min)
            mins.append(chunk_min)
        dist_min = torch.cat(mins, dim=0)
    s_star = float(torch.max(dist_min).item())
    dist_np = dist_min.cpu().numpy()
    dmin = dist_np.min()
    dmax = dist_np.max()
    if dmax - dmin < 1e-8:
        norm = np.zeros_like(dist_np, dtype=np.float32)
    else:
        norm = (dist_np - dmin) / (dmax - dmin)
    seg_small = norm.reshape(h, w).astype(np.float32)
    seg_small_u8 = (seg_small * 255).astype(np.uint8)
    pil_small = Image.fromarray(seg_small_u8)
    pil_up = pil_small.resize(interp_size, resample=Image.BILINEAR)
    seg_up = np.array(pil_up).astype(np.float32) / 255.0
    return seg_up, s_star, dist_np

# ----------------------------
# Visualization helpers
# ----------------------------
def create_enhanced_heatmap(segmap, best_threshold=None, gamma=1.8):
    heat = segmap.copy()
    if best_threshold is None:
        vmin = np.percentile(heat.flatten(), 99)
    else:
        vmin = best_threshold
    vmax = vmin * 2.0 if vmin > 0 else heat.max() or 1.0
    norm = np.clip((heat - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    heat_u8 = (norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    red_boost = (norm ** gamma) * 255.0
    heat_color[..., 0] = np.clip(heat_color[..., 0] + red_boost * 0.7, 0, 255)
    return heat_color.astype(np.uint8), norm

def overlay_image(pil_img, heat_color, alpha=0.45):
    img_np = np.array(pil_img).astype(np.float32)
    h, w = heat_color.shape[:2]
    img_resized = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
    blended = ((1-alpha) * img_resized + alpha * heat_color.astype(np.float32)).astype(np.uint8)
    return blended

# ----------------------------
# Thumbnail widget helper (QWidget with image + filename)
# ----------------------------
from PyQt5.QtWidgets import QVBoxLayout, QSizePolicy, QWidget

THUMB_SIZE = 96  # chosen: 96x96 as requested

def create_thumb_widget(pixmap: QPixmap, filename: str, status: str = "unknown"):
    """
    Returns a QWidget containing image (scaled to THUMB_SIZE) and filename label below.
    We'll use stylesheet on the QLabel (image) to control border color.
    """
    widget = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(2, 2, 2, 2)
    layout.setSpacing(4)
    widget.setLayout(layout)

    lbl_img = QLabel()
    lbl_img.setFixedSize(THUMB_SIZE, THUMB_SIZE)
    lbl_img.setAlignment(Qt.AlignCenter)
    # ensure pixmap scaled to fit inside THUMB_SIZE keeping aspect
    thumb = pixmap.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    lbl_img.setPixmap(thumb)
    lbl_img.setObjectName("thumb_img")
    lbl_img.setStyleSheet(_thumb_style_qss(status))

    lbl_text = QLabel(filename)
    lbl_text.setAlignment(Qt.AlignCenter)
    lbl_text.setWordWrap(False)
    lbl_text.setFixedWidth(THUMB_SIZE + 8)  # let text wrap if needed
    f = QFont()
    f.setPointSize(8)
    lbl_text.setFont(f)
    lbl_text.setStyleSheet("color: #EEEEEE;")
    lbl_text.setContentsMargins(0,0,0,0)

    layout.addWidget(lbl_img, alignment=Qt.AlignCenter)
    layout.addWidget(lbl_text, alignment=Qt.AlignCenter)

    return widget, lbl_img, lbl_text

def _thumb_style_qss(status):
    if status == "unknown":
        border = "#9E9E9E"
    elif status == "ok":
        border = "#2e7d32"
    elif status == "ng":
        border = "#c62828"
    else:
        border = "#9E9E9E"

    return (
        "QLabel#thumb_img {{ "
        f"background: #2b2b2b; border: 1px solid {border}; padding: 2px; }}"
        "QLabel#thumb_img:hover {{ "
        "border-width:2px; }}"
    )


# ----------------------------
# Main PyQt GUI with gallery
# ----------------------------
class PatchCoreApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatchCore - GPU-lite (Gallery 96x96)")
        self.setGeometry(60, 30, 1400, 820)

        # device
        self.device = "cpu"
        self.backbone = PatchCoreBackbone(pretrained=True).to(self.device)

        # memory storage
        self.memory_bank_np = None
        self.best_threshold = None
        self.train_scores = None

        # last results
        self.last_segmap = None
        self.last_overlay = None
        self.last_mask = None
        self.last_score = None

        # test list: track items mapping
        # item.data(Qt.UserRole) -> filepath
        self._init_ui()

    def _init_ui(self):
        main = QHBoxLayout()
        self.setLayout(main)

        # LEFT: controls
        left = QVBoxLayout()
        self.gpu_checkbox = QCheckBox("Use GPU (CUDA for backbone only)")
        self.gpu_checkbox.stateChanged.connect(self.toggle_gpu)
        left.addWidget(self.gpu_checkbox)

        self.btn_add_ok = QPushButton("Add OK Images (Train)")
        self.btn_add_ok.clicked.connect(self.add_ok_images)
        left.addWidget(self.btn_add_ok)

        self.ok_list = QListWidget()
        left.addWidget(self.ok_list, stretch=1)

        self.btn_build = QPushButton("Build Memory Bank (safe)")
        self.btn_build.clicked.connect(self.build_memory)
        left.addWidget(self.btn_build)

        self.progress = QProgressBar()
        self.progress.setRange(0,100)
        left.addWidget(self.progress)

        self.btn_save = QPushButton("Save Memory (.npz)")
        self.btn_save.clicked.connect(self.save_memory)
        left.addWidget(self.btn_save)

        self.btn_load = QPushButton("Load Memory (.npz)")
        self.btn_load.clicked.connect(self.load_memory)
        left.addWidget(self.btn_load)

        left.addStretch()
        main.addLayout(left, 34)

        # RIGHT: images + gallery
        right = QVBoxLayout()
        gb = QGroupBox("Detection")
        gbl = QGridLayout()

        self.lbl_orig = QLabel("Original")
        self.lbl_orig.setFixedSize(520,520)
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("background:#EEE; border:1px solid #444;")

        self.lbl_result = QLabel("Result")
        self.lbl_result.setFixedSize(520,520)
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setStyleSheet("background:#EEE; border:1px solid #444;")

        gbl.addWidget(self.lbl_orig, 0, 0)
        gbl.addWidget(self.lbl_result, 0, 1)
        gb.setLayout(gbl)
        right.addWidget(gb)

        # control row: add test images, show overlay/heat/mask, clear
        ctrl_row = QHBoxLayout()
        self.btn_add_test = QPushButton("+ Add Test Images")
        self.btn_add_test.clicked.connect(self.add_test_images)
        ctrl_row.addWidget(self.btn_add_test)

        self.btn_clear = QPushButton("Clear Test Images")
        self.btn_clear.clicked.connect(self.clear_test_images)
        ctrl_row.addWidget(self.btn_clear)

        self.btn_test = QPushButton("Load Single Test Image")
        self.btn_test.clicked.connect(self.test_image_single_file)
        ctrl_row.addWidget(self.btn_test)

        self.btn_view_overlay = QPushButton("Show Overlay")
        self.btn_view_overlay.clicked.connect(self.show_overlay)
        ctrl_row.addWidget(self.btn_view_overlay)
        self.btn_view_heat = QPushButton("Show Raw Heatmap")
        self.btn_view_heat.clicked.connect(self.show_heatmap)
        ctrl_row.addWidget(self.btn_view_heat)
        self.btn_view_mask = QPushButton("Show Mask")
        self.btn_view_mask.clicked.connect(self.show_mask)
        ctrl_row.addWidget(self.btn_view_mask)
        right.addLayout(ctrl_row)

        # Gallery: horizontal QListWidget in IconMode
        self.test_list = QListWidget()
        self.test_list.setViewMode(QListWidget.IconMode)
        self.test_list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self.test_list.setResizeMode(QListWidget.Adjust)
        self.test_list.setMovement(QListWidget.Static)
        self.test_list.setSpacing(6)
        # fixed height to fit thumbs + filename
        self.test_list.setFixedHeight(THUMB_SIZE + 60)
        # Always show horizontal scrollbar
        self.test_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.test_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.test_list.setFlow(QListWidget.LeftToRight)
        self.test_list.setWrapping(False)
        self.test_list.itemClicked.connect(self.on_test_thumbnail_clicked)
        # dark background
        self.test_list.setStyleSheet("QListWidget { background-color: #2b2b2b; border: none; }")
        right.addWidget(self.test_list)

        # status & score
        self.lbl_status = QLabel("Status: -")
        self.lbl_status.setFixedHeight(36)
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-weight:bold; font-size:16px;")
        right.addWidget(self.lbl_status)

        self.lbl_score = QLabel("Anomaly s*: -")
        self.lbl_score.setStyleSheet("font-size:14px;")
        right.addWidget(self.lbl_score)

        self.lbl_cycle = QLabel("Cycle time: - ms")
        self.lbl_cycle.setStyleSheet("font-size:14px; color:#333;")
        right.addWidget(self.lbl_cycle)

        main.addLayout(right, 66)

    # --------- actions ----------
    def toggle_gpu(self):
        if self.gpu_checkbox.isChecked():
            if torch.cuda.is_available():
                self.device = "cuda"
                QMessageBox.information(self, "GPU", "CUDA is available. Backbone will run on GPU (inference).")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            else:
                QMessageBox.warning(self, "GPU", "CUDA NOT available. Staying on CPU.")
                self.gpu_checkbox.setChecked(False)
                self.device = "cpu"
        else:
            self.device = "cpu"
        self.backbone = PatchCoreBackbone(pretrained=True).to(self.device)

    def add_ok_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select OK images (train)", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        for f in files:
            self.ok_list.addItem(f)

    def build_memory(self):
        n = self.ok_list.count()
        if n == 0:
            QMessageBox.warning(self, "No images", "Please add OK images first.")
            return
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        image_paths = [self.ok_list.item(i).text() for i in range(n)]
        def progress_cb(v):
            self.progress.setValue(v)
            QApplication.processEvents()
        try:
            mem_np, best_thr, train_scores = build_memory_bank_safe(
                backbone=self.backbone,
                image_paths=image_paths,
                device=self.device,
                max_patches=20000,
                progress_callback=progress_cb
            )
        except Exception as e:
            QMessageBox.critical(self, "Build error", str(e))
            return
        self.memory_bank_np = mem_np
        self.best_threshold = best_thr
        self.train_scores = train_scores
        QMessageBox.information(self, "Done", f"Built memory: {mem_np.shape[0]} patches. best_threshold(p99)={best_thr:.6f}")

    def save_memory(self):
        if getattr(self, "memory_bank_np", None) is None:
            QMessageBox.warning(self, "No memory", "Build or load memory first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save memory", "", "NPZ Files (*.npz)")
        if not path:
            return
        np.savez_compressed(path, memory=self.memory_bank_np, best_threshold=np.array([self.best_threshold]) if self.best_threshold is not None else np.array([]))
        QMessageBox.information(self, "Saved", f"Memory saved to {path}")

    def load_memory(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load memory", "", "NPZ Files (*.npz)")
        if not path:
            return
        d = np.load(path)
        self.memory_bank_np = d['memory']
        if 'best_threshold' in d and len(d['best_threshold'])>0:
            self.best_threshold = float(d['best_threshold'][0])
        else:
            self.best_threshold = None
        QMessageBox.information(self, "Loaded", f"Loaded memory ({self.memory_bank_np.shape[0]} patches). best_threshold={self.best_threshold}")

    # ---------- Gallery functions ----------
    def add_test_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select test images", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        for path in files:
            if not os.path.exists(path):
                continue
            # create list item
            item = QListWidgetItem()
            item.setData(Qt.UserRole, path)
            # create pixmap
            pix = QPixmap(path)
            if pix.isNull():
                continue
            # create widget (image + filename)
            widget, lbl_img, lbl_text = create_thumb_widget(pix, os.path.basename(path), status="unknown")
            # store references in item
            item.setSizeHint(widget.sizeHint())
            self.test_list.addItem(item)
            self.test_list.setItemWidget(item, widget)
            item.setData(Qt.UserRole + 1, "unknown")  # status
            # store widget parts for quick access
            item.setData(Qt.UserRole + 2, lbl_img)
            item.setData(Qt.UserRole + 3, lbl_text)

    def clear_test_images(self):
        # clear all thumbnails and reset UI
        self.test_list.clear()
        # reset result displays
        self.last_segmap = None
        self.last_overlay = None
        self.last_mask = None
        self.last_score = None
        self.lbl_orig.setPixmap(QPixmap())
        self.lbl_result.setPixmap(QPixmap())
        self.lbl_status.setText("Status: -")
        self.lbl_status.setStyleSheet("font-weight:bold; font-size:16px;")
        self.lbl_score.setText("Anomaly s*: -")

    def update_thumbnail_status(self, path, status):
        # find item by path and update its widget border
        for i in range(self.test_list.count()):
            it = self.test_list.item(i)
            p = it.data(Qt.UserRole)
            if p == path:
                it.setData(Qt.UserRole + 1, status)
                lbl_img = it.data(Qt.UserRole + 2)
                if isinstance(lbl_img, QLabel):
                    lbl_img.setStyleSheet(_thumb_style_qss(status))
                # update filename color (optional)
                lbl_text = it.data(Qt.UserRole + 3)
                if isinstance(lbl_text, QLabel):
                    if status == "ok":
                        lbl_text.setStyleSheet("color: #A5D6A7;")
                    elif status == "ng":
                        lbl_text.setStyleSheet("color: #EF9A9A;")
                    else:
                        lbl_text.setStyleSheet("color: #EEEEEE;")
                break


    def on_test_thumbnail_clicked(self, item):
        t0 = time.perf_counter()
        path = item.data(Qt.UserRole)
        if not path or not os.path.exists(path):
            return
        pil = Image.open(path).convert("RGB")
        # show orig (thumbnail scaled to 520)
        try:
            self.lbl_orig.setPixmap(pil_to_qpixmap(pil.resize((520,520))))
        except Exception:
            # fallback - convert to QPixmap via QImage
            pix = QPixmap(path).scaled(520,520, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_orig.setPixmap(pix)
        # ensure memory exists
        if getattr(self, "memory_bank_np", None) is None:
            QMessageBox.warning(self, "No memory", "Please build or load memory first.")
            return
        # run detect
        try:
            segmap_up, s_star, per_patch = detect_image_safe(
                backbone=self.backbone,
                memory_bank_np=self.memory_bank_np,
                pil_img=pil,
                device=self.device,
                interp_size=(224,224)
            )
        except Exception as e:
            QMessageBox.critical(self, "Detect error", str(e))
            return
        self.last_segmap = segmap_up
        self.last_score = s_star
        thr = self.best_threshold if getattr(self, "best_threshold", None) is not None else float(np.percentile(self.last_segmap.flatten(), 99))
        heat_color, norm_map = create_enhanced_heatmap(self.last_segmap, best_threshold=thr, gamma=1.9)
        seg_mask = (self.last_segmap > thr * 1.25).astype(np.uint8) * 255
        orig_w, orig_h = pil.size
        heat_up = cv2.resize(heat_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_up = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay = overlay_image(pil, heat_up, alpha=0.45)
        norm_up = cv2.resize(norm_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        red_boost = (norm_up ** 2.0) * 255.0
        overlay[..., 0] = np.clip(overlay[..., 0] + red_boost * 0.7, 0, 255)
        overlay = overlay.astype(np.uint8)
        self.last_overlay = overlay
        self.last_mask = mask_up
        # update UI
        self.update_status(s_star, thr)
        self.show_overlay()
        self.lbl_score.setText(f"s*: {s_star:.6f}    thr: {thr:.6f}    s*/thr: {s_star/(thr+1e-8):.3f}")
        # set thumbnail border OK/NG
        status = "ok" if s_star <= thr else "ng"
        self.update_thumbnail_status(path, status)

        t1 = time.perf_counter()
        cycle_ms = (t1 - t0) * 1000.0
        if status == "ng":
            self.lbl_cycle.setStyleSheet("font-size:14px; color:#c62828;")
        else:
            self.lbl_cycle.setStyleSheet("font-size:14px; color:#2e7d32;")

        self.lbl_cycle.setText(f"Cycle time: {cycle_ms:.1f} ms")

    # ---------- single-file test (kept for compatibility) ----------
    def test_image_single_file(self):
        if getattr(self, "memory_bank_np", None) is None:
            QMessageBox.warning(self, "No memory", "Please build or load memory first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select test image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        # If not already in gallery, add and then simulate click
        found = None
        for i in range(self.test_list.count()):
            it = self.test_list.item(i)
            if it.data(Qt.UserRole) == path:
                found = it
                break
        if found is None:
            pix = QPixmap(path)
            widget, lbl_img, lbl_text = create_thumb_widget(pix, os.path.basename(path), status="unknown")
            item = QListWidgetItem()
            item.setData(Qt.UserRole, path)
            item.setSizeHint(widget.sizeHint())
            self.test_list.addItem(item)
            self.test_list.setItemWidget(item, widget)
            item.setData(Qt.UserRole + 1, "unknown")
            item.setData(Qt.UserRole + 2, lbl_img)
            item.setData(Qt.UserRole + 3, lbl_text)
            found = item
        # trigger detect
        self.on_test_thumbnail_clicked(found)

    # ---------- visualization toggles ----------
    def update_status(self, s_star, thr):
        if thr is None:
            thr = float(np.percentile(self.last_segmap.flatten(), 99))
        if s_star > thr:
            self.lbl_status.setText("Status: NG")
            self.lbl_status.setStyleSheet("background-color:#c62828; color:white; font-weight:bold; font-size:16px;")
        else:
            self.lbl_status.setText("Status: OK")
            self.lbl_status.setStyleSheet("background-color:#2e7d32; color:white; font-weight:bold; font-size:16px;")

    def show_overlay(self):
        if self.last_overlay is None:
            return
        pil = Image.fromarray(self.last_overlay).resize((520,520))
        self.lbl_result.setPixmap(pil_to_qpixmap(pil))

    def show_heatmap(self):
        if self.last_segmap is None:
            return
        thr = self.best_threshold if getattr(self, "best_threshold", None) is not None else float(np.percentile(self.last_segmap.flatten(), 99))
        heat_color, _ = create_enhanced_heatmap(self.last_segmap, best_threshold=thr, gamma=1.9)
        orig_w, orig_h = self.lbl_orig.pixmap().width(), self.lbl_orig.pixmap().height()
        heat_up = cv2.resize(heat_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(heat_up).resize((520,520))
        self.lbl_result.setPixmap(pil_to_qpixmap(pil))

    def show_mask(self):
        if self.last_mask is None:
            return
        mask = self.last_mask.copy()
        w = self.lbl_orig.pixmap().width(); h = self.lbl_orig.pixmap().height()
        mask_small = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_rgb = np.stack([mask_small, np.zeros_like(mask_small), np.zeros_like(mask_small)], axis=2)
        orig_pix = self.lbl_orig.pixmap()
        if orig_pix is None:
            return
        orig_img = orig_pix.toImage()
        orig_w2 = orig_img.width(); orig_h2 = orig_img.height()
        ptr = orig_img.bits()
        ptr.setsize(orig_h2*orig_w2*4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((orig_h2, orig_w2, 4))
        orig_np = arr[..., :3].copy()
        blended = np.clip(0.6*orig_np + 0.4*mask_rgb, 0, 255).astype(np.uint8)
        pil = Image.fromarray(blended).resize((520,520))
        self.lbl_result.setPixmap(pil_to_qpixmap(pil))


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PatchCoreApp()
    win.show()
    sys.exit(app.exec_())




