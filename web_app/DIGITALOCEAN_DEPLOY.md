# DigitalOcean Droplet Deployment Guide

## 1. Create a Droplet

1. Go to [DigitalOcean](https://cloud.digitalocean.com/)
2. Create a new Droplet:
   - **Image**: Ubuntu 22.04 LTS
   - **Size**: Basic → Regular → 2GB RAM / 1 CPU (minimum for ML inference)
   - **Region**: Choose closest to your users
   - **Authentication**: SSH Key (recommended) or Password

## 2. Connect to Your Droplet

```bash
ssh root@<YOUR_DROPLET_IP>
```

## 3. Initial Server Setup

```bash
# Update system
apt update && apt upgrade -y

# Install Python and dependencies
apt install -y python3 python3-pip python3-venv git

# Install Docker (optional, for containerized deployment)
apt install -y docker.io docker-compose
systemctl enable docker
systemctl start docker
```

## 4. Clone and Setup Application

```bash
# Clone repository
git clone <YOUR_REPO_URL> /opt/webapp
cd /opt/webapp/webapp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 5. Configure Firewall

```bash
# Allow SSH and application port
ufw allow 22
ufw allow 8000
ufw enable
```

## 6. Run the Application

### Option A: Direct Python (Development)

```bash
cd /opt/webapp/webapp
source venv/bin/activate
python app.py
```

### Option B: Docker (Recommended for Production)

```bash
cd /opt/webapp/webapp
docker build -t brain-tumor-api .
docker run -d -p 8000:8000 --name brain-tumor-api brain-tumor-api
```

### Option C: Systemd Service (Production)

Create service file:

```bash
cat > /etc/systemd/system/brain-tumor-api.service << EOF
[Unit]
Description=Brain Tumor Segmentation API
After=network.target

[Service]
User=root
WorkingDirectory=/opt/webapp/webapp
Environment="PATH=/opt/webapp/webapp/venv/bin"
ExecStart=/opt/webapp/webapp/venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start:

```bash
systemctl daemon-reload
systemctl enable brain-tumor-api
systemctl start brain-tumor-api
systemctl status brain-tumor-api
```

## 7. Update Mobile App

In `mobile/.env`:

```env
API_BASE_URL=http://<YOUR_DROPLET_IP>:8000
```

## 8. Optional: Setup Domain & SSL

```bash
# Install Nginx
apt install -y nginx

# Install Certbot for SSL
apt install -y certbot python3-certbot-nginx

# Configure Nginx (create /etc/nginx/sites-available/brain-tumor-api)
# Get SSL certificate
certbot --nginx -d yourdomain.com
```

## Useful Commands

```bash
# View logs
journalctl -u brain-tumor-api -f

# Restart service
systemctl restart brain-tumor-api

# Check Docker logs
docker logs -f brain-tumor-api
```
