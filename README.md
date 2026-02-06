# 🧠 BioMedIAMBZ - Brain Tumor Segmentation with MedNeXt

<p align="center">
  <img src="images/img_0.gif" width="200" />
  <img src="images/img_1.gif" width="200" />
  <img src="images/img_2.gif" width="200" />
  <img src="images/img_3.gif" width="200" />
</p>

<p align="center">
  <strong>BraTS 2024 SSA & Pediatrics Challenge Solution</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/MONAI-1.3+-green" alt="MONAI" />
  <img src="https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?logo=kaggle" alt="Kaggle" />
</p>

---

## 📋 Overview

This repository contains our solution for the **BraTS 2024 Brain Tumor Segmentation Challenge**, focusing on Sub-Saharan Africa (SSA) and Pediatric tumor datasets. We leverage the **MedNeXt** architecture, a state-of-the-art 3D medical image segmentation model that combines the efficiency of ConvNeXt with medical imaging-specific optimizations.

### 🎯 Challenge Tasks

- **BraTS-SSA**: Brain tumor segmentation for Sub-Saharan African patients
- **BraTS-PED**: Pediatric brain tumor segmentation

### 🏆 Key Features

- **MedNeXt Architecture**: State-of-the-art 3D CNN outperforming transformer-based models
- **Deep Supervision**: Enhanced training with multi-scale loss computation
- **Model Souping**: Ensemble technique for improved generalization
- **Multi-platform Deployment**: Web and mobile applications for accessibility

---

## 🚀 Kaggle Notebooks (Main Codebase)

Our primary training and inference pipelines are developed as Kaggle notebooks for easy reproducibility with free GPU resources:

### Training Notebooks

| Notebook                                                    | Description               | Architecture               |
| ----------------------------------------------------------- | ------------------------- | -------------------------- |
| [**MedNeXt.ipynb**](kaggle_notebooks/MedNeXt.ipynb)         | Main training pipeline    | MedNeXt-B                  |
| [**new-mednext.ipynb**](kaggle_notebooks/new-mednext.ipynb) | Enhanced MedNeXt training | MedNeXt with optimizations |
| [**SegRes-Net.ipynb**](kaggle_notebooks/SegRes-Net.ipynb)   | SegResNet baseline        | SegResNet                  |
| [**U-net.ipynb**](kaggle_notebooks/U-net.ipynb)             | U-Net baseline            | Classic U-Net              |

### Inference Notebooks

| Notebook                                                                                     | Description                 |
| -------------------------------------------------------------------------------------------- | --------------------------- |
| [**mednext-inference.ipynb**](kaggle_notebooks/mednext-inference.ipynb)                      | Standard MedNeXt inference  |
| [**mednext-enahnced-inference.ipynb**](kaggle_notebooks/mednext-enahnced-inference%20.ipynb) | Enhanced inference with TTA |
| [**SegRes-Net_inference.ipynb**](kaggle_notebooks/SegRes-Net_inference.ipynb)                | SegResNet inference         |
| [**U-net_inference.ipynb**](kaggle_notebooks/U-net_inference.ipynb)                          | U-Net inference             |

> **💡 Tip**: Run these notebooks directly on Kaggle with P100/T4 GPUs for free!

---

## 🏗️ Project Structure

```
├── 📁 kaggle_notebooks/          # 🔥 Main training & inference code
│   ├── MedNeXt.ipynb
│   ├── new-mednext.ipynb
│   ├── mednext-inference.ipynb
│   └── ...
├── 📁 biomedmbz_glioma/          # Core PyTorch Lightning module
│   ├── dataset.py                # BraTS dataset loaders
│   ├── pl_module.py              # Training module with deep supervision
│   ├── transforms.py             # Data augmentation pipeline
│   ├── metrics.py                # Dice score metrics
│   ├── loss.py                   # Combined loss functions
│   ├── inference.py              # Inference utilities
│   └── postprocessing.py         # Post-processing (small ET removal)
├── 📁 nnunet_mednext/            # MedNeXt architecture implementation
│   ├── network_architecture/     # Model definitions
│   ├── training/                 # nnU-Net style training
│   └── inference/                # Sliding window inference
├── 📁 web_app/                   # 🌐 FastAPI web application
├── 📁 mobile_app/                # 📱 React Native mobile app
├── 📁 model_soup/                # Model ensemble utilities
├── 📁 presentation/              # Hackathon presentation materials
├── mednext_train.py              # Local training script
├── preprocessing.py              # Data preprocessing CLI
├── souping.py                    # Model souping script
└── visualize_3d.py               # 3D visualization utilities
```

---

## 🔬 MedNeXt Architecture

We chose **MedNeXt** over transformer-based architectures for several key reasons:

- **🎯 Superior Performance**: Outperforms Swin-UNETR and other transformers on BraTS benchmarks
- **⚡ Computational Efficiency**: 4x faster training with lower memory footprint
- **🔄 ConvNeXt Innovations**: Incorporates modern design principles (larger kernels, LayerNorm, GELU)
- **📊 Deep Supervision**: Multi-scale loss for better gradient flow

```
MedNeXt-B Configuration:
├── Kernel Size: 3x3x3
├── Deep Supervision: Enabled (4 levels)
├── ROI Size: 128×128×128 (full) / 64×64×64 (low VRAM)
├── Optimizer: AdamW with ScheduleFree
└── Loss: DiceCE + Deep Supervision
```

---

## 🖥️ Local Development

### Prerequisites

- Python 3.10+
- CUDA 11.8+ compatible GPU (8GB+ VRAM recommended)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/KarimmYasser/braTs-ai-hackathon-ODCxINSTANT.git
cd braTs-ai-hackathon-ODCxINSTANT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

1. **Preprocess the data**:

```bash
python preprocessing.py --input /path/to/BraTS2024 --output ./preprocessed_data
```

2. **Configure training** in `train_args.json`:

```json
{
  "fold": 0,
  "max_epochs": 50,
  "batch_size": 1,
  "mednext_size": "B",
  "roi_x": 64,
  "roi_y": 64,
  "roi_z": 64,
  "deep_sup": true,
  "lr": 0.002
}
```

3. **Start training**:

```bash
python mednext_train.py
```

### Inference

Use the Kaggle inference notebooks or run locally:

```bash
python -m biomedmbz_glioma.inference --model checkpoints/best.pt --input /path/to/test
```

---

## 🌐 Web Application

A modern FastAPI-based web application for brain tumor segmentation.

<p align="center">
  <strong>Features:</strong> Multi-model Support | 2D Slice Viewer | 3D Multi-View | Real-time Processing
</p>

```bash
cd web_app
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

📚 **Deployment Guides**: [Azure](web_app/AZURE_DEPLOY.md) | [DigitalOcean](web_app/DIGITALOCEAN_DEPLOY.md)

➡️ See [web_app/README.md](web_app/README.md) for details.

---

## 📱 Mobile Application

A React Native / Expo mobile app for brain tumor segmentation visualization.

<p align="center">
  <strong>Features:</strong> Cross-Platform (iOS/Android) | 2D Slice Navigation | 3D Multi-View | Tumor Statistics
</p>

```bash
cd mobile_app
npm install
npx expo start
```

➡️ See [mobile_app/README.md](mobile_app/README.md) for details.

---

## 📊 Results

### Tumor Classes

| Label | Class   | Description       |
| ----- | ------- | ----------------- |
| 1     | **NCR** | Necrotic Core     |
| 2     | **ED**  | Peritumoral Edema |
| 3     | **ET**  | Enhancing Tumor   |

### Evaluation Metrics

- **Dice Score (DSC)**: Overlap-based metric
- **Hausdorff Distance (HD95)**: Surface distance metric

---

## 📂 Trained Models

Pre-trained model weights are available:

| Model                             | Architecture | Dataset    |
| --------------------------------- | ------------ | ---------- |
| `models/mednext-model.pt`         | MedNeXt-B    | BraTS 2024 |
| `models/classical-unet-model.pth` | U-Net        | BraTS 2024 |

---

## 🤝 Team BioMedIAMBZ

This project was developed for the **ODC x INSTANT AI Hackathon**.

---

## 📚 References

- [MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation](https://arxiv.org/abs/2303.09975)
- [nnU-Net: Self-adapting Framework for Medical Image Segmentation](https://github.com/MIC-DKFZ/nnUNet)
- [BraTS Challenge 2024](https://www.synapse.org/brats)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
