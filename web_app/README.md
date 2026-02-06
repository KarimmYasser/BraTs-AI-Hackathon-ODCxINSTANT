# 🧠 Brain Tumor Segmentation - Web App

A modern web application for brain tumor segmentation using deep learning models (U-Net, SegResNet, MedNeXt).

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

## Features

- 🔬 **Multi-model Support** - U-Net, SegResNet, MedNeXt architectures
- 🖼️ **2D Slice Viewer** - Navigate through brain MRI slices
- 🎯 **3D Multi-View** - Axial, coronal, and sagittal visualization
- 📊 **Tumor Statistics** - NCR, ED, and ET voxel counts
- 🎨 **Dark Theme** - Modern, eye-friendly interface

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

## Usage

1. **Upload** - Select 4 MRI modalities (T1, T1ce, T2, FLAIR) as `.nii.gz` files
2. **Select Model** - Choose segmentation model
3. **View Results** - Explore slices or switch to 3D multi-view

## API Endpoints

| Endpoint                    | Method | Description      |
| --------------------------- | ------ | ---------------- |
| `/health`                   | GET    | Health check     |
| `/upload`                   | POST   | Upload MRI files |
| `/segment/{session_id}`     | POST   | Run segmentation |
| `/slice/{session_id}/{idx}` | GET    | Get 2D slice     |
| `/multiview/{session_id}`   | GET    | Get 3D views     |

## Azure Deployment

See [AZURE_DEPLOY.md](./AZURE_DEPLOY.md) for deployment instructions.

## Project Structure

```
webapp/
├── app.py              # FastAPI application
├── models.py           # Model loading & inference
├── utils.py            # Image processing utilities
├── static/
│   ├── index.html      # Frontend HTML
│   ├── styles.css      # Dark theme styles
│   └── app.js          # Frontend logic
└── scripts/
    └── azure-setup.sh  # Azure VM setup script
```

## License

MIT
