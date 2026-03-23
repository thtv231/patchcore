````# 🔍 PatchCore – Visual Anomaly Detection GUI

<p align="center">
  <img src="images/demo_ng_screen.png" width="700" alt="PatchCore Demo"/>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=Iz6ui04gJaE">
    <img src="https://img.shields.io/badge/Demo-YouTube-red?logo=youtube" alt="YouTube Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

Ứng dụng phát hiện lỗi ngoại quan (anomaly detection) trong công nghiệp sử dụng phương pháp **PatchCore** kết hợp giao diện đồ họa PyQt5. Hệ thống hoạt động theo hướng **học tự giám sát (self-supervised)** — chỉ cần ảnh bình thường để huấn luyện, không cần ảnh lỗi.

---

## 📺 Demo

👉 [Xem demo trên YouTube](https://www.youtube.com/watch?v=Iz6ui04gJaE)

---

## ✨ Tính năng

- **Không cần GPU để train** — Memory Bank được xây dựng từ CPU, deploy ngay lập tức
- **Gallery 96×96** — Duyệt nhiều ảnh test cùng lúc, click để detect từng ảnh
- **Heatmap trực quan** — Hiển thị vùng bất thường dạng overlay màu nhiệt
- **3 chế độ xem** — Overlay / Raw Heatmap / Mask
- **Chỉ số rõ ràng** — Hiển thị `s*`, `thr`, `s*/thr` và cycle time (ms)
- **Lưu / Load Memory Bank** — Lưu file `.npz` để tái sử dụng không cần build lại
- **Coreset 10%** — Giảm kích thước memory bank, tăng tốc độ inference

---

## 🖥️ Giao diện

| Trạng thái | Mô tả |
|---|---|
| 🟥 **Status: NG** | `s* > thr` — Phát hiện bất thường |
| 🟩 **Status: OK** | `s* ≤ thr` — Ảnh bình thường |

<table>
  <tr>
    <td align="center"><b>NG – Phát hiện lỗi</b></td>
    <td align="center"><b>OK – Bình thường</b></td>
  </tr>
  <tr>
    <td><img src="images/demo_ng_screen.png" width="380"/></td>
    <td><img src="images/demo_ok_screen.png" width="380"/></td>
  </tr>
</table>

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

- Python >= 3.8
- RAM >= 4GB
- GPU (tùy chọn, CUDA)

### Cài thư viện

```bash
pip install pyqt5 torch torchvision numpy pillow opencv-python
```

### Chạy ứng dụng

```bash
python app.py
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1 — Thêm ảnh train (bình thường)

Nhấn **"Add OK Images (Train)"** → chọn thư mục chứa ảnh bình thường.

### Bước 2 — Build Memory Bank

Nhấn **"Build Memory Bank (safe)"** → chờ thanh tiến trình đến 100%.

> 💡 Có thể lưu lại bằng **"Save Memory (.npz)"** để dùng lần sau mà không cần build lại.

### Bước 3 — Load ảnh test

Nhấn **"+ Add Test Images"** → chọn các ảnh cần kiểm tra.

### Bước 4 — Detect

**Click vào thumbnail** bất kỳ trong gallery để chạy phát hiện bất thường.

- Kết quả hiển thị ngay: heatmap, anomaly score, và trạng thái **OK / NG**
- Thumbnail tự đổi viền: 🟢 xanh (OK) / 🔴 đỏ (NG)

---

## 🧠 Kiến trúc kỹ thuật

```
Ảnh train (bình thường)
        │
        ▼
  ResNet50 Backbone
  (layer2 + layer3)
        │
        ▼
  Patch Features [N × C]
        │
        ▼
  Coreset Sampling (10%)
        │
        ▼
  Memory Bank (.npz)
        
        ┄┄┄┄┄ Inference ┄┄┄┄┄
        
  Ảnh test → Patch Features
        │
        ▼
  NN Distance đến Memory Bank
        │
        ▼
  s* = max(min distance)
        │
     s* > thr?
    /         \
  NG           OK
```

### Tính toán Anomaly Score

$$s^* = \max_{i} \min_{m \in \mathcal{M}} \| p_i - m \|_2$$

Ngưỡng `thr` được xác định từ **phân vị 99%** của anomaly scores trên tập train.

---

## 📁 Cấu trúc thư mục

```
patchcore/
├── app.py              # Ứng dụng chính
├── README.md
├── data/
│   └── carpet/
│       ├── train/
│       │   └── good/   # Ảnh bình thường để train
│       └── test/       # Ảnh test
├── memory/             # Lưu file .npz memory bank
└── images/             # Ảnh demo cho README
```

---

## 📊 Kết quả thực nghiệm

Dataset: **MVTec AD – Carpet**

| Chỉ số | Giá trị |
|---|---|
| Backbone | ResNet50 (ImageNet pretrained) |
| Coreset ratio | 10% |
| Cycle time | ~538–646 ms/ảnh (CPU) |
| Ngưỡng `thr` | 0.7029 (phân vị 99%) |

| Ảnh | Anomaly Score `s*` | Kết quả |
|---|---|---|
| Ảnh bình thường | 0.6188 | ✅ OK |
| Ảnh có lỗi | 0.8731 | ❌ NG |
---
## 📚 Tài liệu tham khảo

- [PatchCore: Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265) — Roth et al., CVPR 2022
- [CutPaste: Self-Supervised Learning for Anomaly Detection](https://arxiv.org/abs/2104.04015) — Li et al., 2021
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 📄 License

MIT License © 2025 – Nhóm 1_AI_NC
````
