# Brain Tumor Segmentation - Azure Deployment

## Azure VM Deployment (Recommended for Demo)

### Step 1: Create Azure VM with GPU

```bash
# Login to Azure CLI
az login

# Create resource group
az group create --name brain-tumor-demo --location eastus

# Create VM with GPU (NC6 - cheapest GPU option ~$0.90/hour)
az vm create \
  --resource-group brain-tumor-demo \
  --name brain-tumor-vm \
  --image Canonical:0001-com-ubuntu-server-focal:20_04-lts-gen2:latest \
  --size Standard_NC6 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Open port 8000 for the web app
az vm open-port --resource-group brain-tumor-demo --name brain-tumor-vm --port 8000
```

### Step 2: SSH into VM and Setup

```bash
# Get the public IP
az vm show -d -g brain-tumor-demo -n brain-tumor-vm --query publicIps -o tsv

# SSH into the VM
ssh azureuser@<PUBLIC_IP>
```

### Step 3: Install Dependencies on VM

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Reboot to load drivers
sudo reboot

# After reboot, verify GPU
nvidia-smi

# Install Python and pip
sudo apt install -y python3-pip python3-venv git

# Clone your repo (after pushing to GitHub)
git clone https://github.com/YOUR_USERNAME/brain-tumor-webapp.git
cd brain-tumor-webapp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
# Run with uvicorn (in background with nohup)
nohup python -m uvicorn app:app --host 0.0.0.0 --port 8000 &

# Or use screen for persistence
screen -S webapp
python -m uvicorn app:app --host 0.0.0.0 --port 8000
# Press Ctrl+A, then D to detach
```

### Step 5: Access Your App

Open in browser: `http://<PUBLIC_IP>:8000`

---

## Cost Management Tips

| VM Size | GPU        | Cost/Hour | $100 Runtime |
| ------- | ---------- | --------- | ------------ |
| NC6     | Tesla K80  | ~$0.90    | ~111 hours   |
| NC6s_v3 | Tesla V100 | ~$3.06    | ~32 hours    |

### To Stop/Start VM (Save Money)

```bash
# Stop VM when not demoing (stops billing)
az vm deallocate --resource-group brain-tumor-demo --name brain-tumor-vm

# Start VM when needed
az vm start --resource-group brain-tumor-demo --name brain-tumor-vm
```

### Clean Up When Done

```bash
# Delete everything to stop all charges
az group delete --name brain-tumor-demo --yes --no-wait
```

---

## Alternative: Docker Deployment

If you prefer Docker, use the included Dockerfile:

```bash
# Build image
docker build -t brain-tumor-app .

# Run with GPU support
docker run --gpus all -p 8000:8000 brain-tumor-app
```
