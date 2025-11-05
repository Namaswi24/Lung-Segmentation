# 🫁 Lung Segmentation using Medical Transformer (MedT)

Automated **lung segmentation** from chest X-ray images using a **Medical Transformer (MedT)** deep learning model.  
This project demonstrates the use of transformer-based architectures for **medical image segmentation**, achieving superior results compared to traditional CNN models like U-Net or SegNet.

---

## 👩‍💻 Team Members
- **Janapareddy Vidya Varshini** – 230041013  
- **Korubilli Vaishnavi** – 230041016  
- **Mullapudi Namaswi** – 230041023  

📂 **GitHub Repository:** [https://github.com/Namaswi24/Lung-Segmentation](https://github.com/Namaswi24/Lung-Segmentation)

---

## 🧩 Problem Statement

> Develop an automated system to segment lung regions from chest X-ray (CXR) images using the **Medical Transformer (MedT)** model.  
> The goal is to build an accurate and efficient segmentation model capable of delineating lung regions with minimal supervision.

Lung segmentation is a crucial preprocessing step in detecting and analyzing pulmonary diseases.  
Manual segmentation is time-consuming and prone to human error — hence, an automated deep-learning solution is proposed.

---

## 📁 Dataset Structure

### Dataset Info
- Total images: **800**
- Matched image-mask pairs: **704**
- Image dimensions: 2300–3000 px range (resized to 128×128)
- Data split:
  - **Train:** 70%
  - **Validation:** 15%
  - **Test:** 15%

---

## ⚙️ Data Preprocessing & Augmentation

Implemented through the `ChestXrayDataset` class:
- **Resize:** All images & masks → 128×128  
- **Binarize masks:** Pixel >127 → 1 (lung), else 0  
- **Normalize images:** Using ImageNet mean & std  
- **Augmentation (for training):**
  - Random horizontal flips
  - Random brightness/contrast (0.8–1.2)
  - Geometrically consistent image-mask transforms  
- **Conversion:** NumPy → PyTorch tensors (C, H, W)

---

## 🧠 Model Architecture — Medical Transformer (MedT)

MedT addresses the limitations of CNNs (poor long-range context) and vanilla Transformers (poor local detail).

### 🧩 Key Components
1. **Gated Axial Attention (GAA):**
   - Decomposes 2D attention into two 1D attentions (height & width).
   - Introduces *learnable gates* for positional encoding.
   - Efficient and ideal for medical imaging tasks.

2. **Local-Global (LoGo) Training Strategy:**
   - **Global Branch:** Operates on a downsampled image (128×128 → 32×32) for contextual understanding.  
   - **Local Branch:** Processes small 4×4 patches for detailed local features.  
   - Combined outputs capture both fine and global context.

---

## ⚙️ Experimental Setup

| Parameter | Value |
|------------|--------|
| Framework | PyTorch |
| Device | CUDA (GPU) if available |
| Image Size | 128×128 |
| Epochs | 40 |
| Batch Size | 4 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 10) |
| Loss | Binary Cross-Entropy (BCE) + Dice Loss |
| Metrics | Accuracy, Precision, Recall, IoU, Dice Score |

---

## 🧪 Results & Observations

- Best model achieved after **19th epoch**
- Evaluated on the **test set**
- High **Dice coefficient** and **IoU** confirming segmentation quality
- Model generalized well on unseen X-rays

| Metric | Description |
|---------|--------------|
| **Dice Coefficient (F1)** | 2TP / (2TP + FP + FN) — measures overlap |
| **IoU (Intersection over Union)** | TP / (TP + FP + FN) |
| **Precision** | TP / (TP + FP) — avoids over-segmentation |
| **Recall** | TP / (TP + FN) — measures detection completeness |

---

## 🖥️ Web Application Deployment

A **Flask-based web interface** was built to make the segmentation model accessible.

### 🌐 Features
- Upload chest X-ray image (`.png`)
- Model performs segmentation and returns mask
- Displays original + segmented output side by side
- Responsive UI built with **HTML, CSS, JavaScript**
- Backend uses **Flask** and pre-trained MedT model

### 🧭 Run the Web App
```bash
# Clone the repository
git clone https://github.com/Namaswi24/Lung-Segmentation.git
cd Lung-Segmentation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
