# Deployment Guide - CloudAI Analytics Team

**Author:** Abdul Salam Aldabik

This guide covers the complete deployment pipeline for both ML applications (UK Housing Prices + UK Electricity Demand).

---

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Local Development](#local-development)
3. [CI/CD Pipeline](#cicd-pipeline)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Hosting Options](#cloud-hosting-options)
6. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🔄 Pipeline Overview

### Automated ML Pipeline

Our GitHub Actions pipeline automatically:
- ✅ Retrains models when new data or code is pushed
- ✅ Validates model performance
- ✅ Saves updated models to repository
- ✅ Triggers deployment notifications

**Trigger:** Push to `main` branch with changes in:
- `Dataset_1_UK_Housing/Code/**`
- `Dataset_2_UK_Historic_Electricity_Demand_Data/Code/**`
- Any data files

**Note:** Commits made by the pipeline include `[skip ci]` to prevent infinite loops.

---

## 💻 Local Development

### Option 1: Python Virtual Environment

```powershell
# Clone repository
git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
cd Machine-Learning-Project-TM-2025

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run Housing app
cd Dataset_1_UK_Housing/Code
streamlit run streamlit_app.py

# Run Electricity app (in new terminal)
cd Dataset_2_UK_Historic_Electricity_Demand_Data/Code
streamlit run streamlit_app.py
```

### Option 2: Docker Compose (Recommended)

```powershell
# Build and run both apps
docker-compose up --build

# Access applications:
# Housing: http://localhost:8501
# Electricity: http://localhost:8502

# Stop containers
docker-compose down
```

---

## 🤖 CI/CD Pipeline

### GitHub Actions Workflow

File: `.github/workflows/ml_pipeline.yml`

**Jobs:**

1. **retrain-housing-model**
   - Trains Ridge regression on latest housing data
   - Saves `housing_ridge_pipeline.pkl` + scaler
   - Commits models back to repository

2. **retrain-electricity-model**
   - Trains XGBoost on latest electricity data
   - Saves `electricity_xgboost_pipeline.pkl`
   - Commits model back to repository

3. **deploy-notification**
   - Sends notification when retraining completes
   - Triggers deployment workflows (if configured)

### Manual Trigger

```powershell
# Via GitHub web interface:
# Actions → ML Model Training Pipeline → Run workflow

# Or push changes to main branch
git add .
git commit -m "Update training data"
git push origin main
```

---

## 🐳 Docker Deployment

### Build Individual Containers

```powershell
# Housing app
docker build -f Dockerfile.housing -t uk-housing-app .
docker run -p 8501:8501 uk-housing-app

# Electricity app
docker build -f Dockerfile.electricity -t uk-electricity-app .
docker run -p 8502:8502 uk-electricity-app
```

### Production Deployment

```powershell
# Build in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Update and redeploy
git pull origin main
docker-compose up --build -d
```

---

## ☁️ Cloud Hosting Options

### Option 1: Oracle Cloud (Free Tier)

**Setup Steps:**

1. **Create Oracle Cloud Account**
   - Visit: https://www.oracle.com/cloud/free/
   - Free tier includes: 2 AMD VMs, 200GB storage

2. **Launch Compute Instance**
   ```bash
   # SSH into instance
   ssh ubuntu@<your-instance-ip>
   
   # Install Docker
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo usermod -aG docker $USER
   
   # Clone repository
   git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
   cd Machine-Learning-Project-TM-2025
   
   # Deploy
   docker-compose up -d
   ```

3. **Configure Firewall**
   - Open ports 8501 and 8502 in Oracle Cloud security list
   - Access: `http://<your-instance-ip>:8501`

### Option 2: AWS EC2

```bash
# Launch t2.micro (Free tier eligible)
# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Deploy apps
git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
cd Machine-Learning-Project-TM-2025
docker-compose up -d
```

### Option 3: Raspberry Pi (Home Hosting)

```bash
# Install Docker on Raspberry Pi OS
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker pi

# Clone and deploy
git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
cd Machine-Learning-Project-TM-2025
docker-compose up -d

# Setup port forwarding on router for ports 8501, 8502
# Use dynamic DNS service (e.g., No-IP) for stable URL
```

### Option 4: Streamlit Community Cloud (Easiest)

```powershell
# 1. Push code to GitHub (already done)
# 2. Visit: https://share.streamlit.io/
# 3. Connect GitHub account
# 4. Deploy apps:
#    - App 1: Dataset_1_UK_Housing/Code/streamlit_app.py
#    - App 2: Dataset_2_UK_Historic_Electricity_Demand_Data/Code/streamlit_app.py
# 5. Get public URLs (e.g., https://yourapp.streamlit.app)
```

**Limitation:** Streamlit Cloud may have memory limits for large datasets.

---

## 📊 Monitoring & Maintenance

### Health Checks

Both Docker containers include health checks:

```powershell
# Check container status
docker ps

# View health status
docker inspect uk-housing-predictor | grep -A 10 Health
docker inspect uk-electricity-forecaster | grep -A 10 Health
```

### Model Retraining Schedule

**Recommended Schedule:**

- **Housing Model:** Retrain monthly (when new property sales data available)
- **Electricity Model:** Retrain weekly (new demand data daily from NESO)

**Setup Automated Retraining:**

```yaml
# Add to .github/workflows/ml_pipeline.yml
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM
```

### Logs

```powershell
# Docker logs
docker-compose logs -f housing-app
docker-compose logs -f electricity-app

# Filter by time
docker-compose logs --since 1h electricity-app
```

### Updates

```powershell
# Pull latest code
git pull origin main

# Rebuild and redeploy
docker-compose down
docker-compose up --build -d
```

---

## 🔐 Security Considerations

1. **Environment Variables**
   - Store API keys in `.env` file (not committed to git)
   - Add `.env` to `.gitignore`

2. **HTTPS**
   - Use reverse proxy (nginx) with SSL certificate
   - Let's Encrypt for free SSL

3. **Access Control**
   - Implement authentication if needed
   - Use Streamlit's built-in password protection

---

## 🚀 Quick Start Commands

```powershell
# Local testing
streamlit run Dataset_1_UK_Housing/Code/streamlit_app.py

# Docker deployment
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Full cleanup (remove images)
docker-compose down --rmi all
```

---

## 📞 Troubleshooting

**Issue: Port already in use**
```powershell
# Change ports in docker-compose.yml
ports:
  - "8503:8501"  # Instead of 8501:8501
```

**Issue: Container crashes**
```powershell
# Check logs
docker logs uk-housing-predictor

# Rebuild
docker-compose up --build
```

**Issue: Model files not found**
```powershell
# Ensure models exist
ls Dataset_1_UK_Housing/Code/Models/
ls Dataset_2_UK_Historic_Electricity_Demand_Data/Code/*.pkl

# Retrain if needed
python train_models.py
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud)
- [Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)

---

**Team:** CloudAI Analytics Team  
**Members:** Jo Naulaerts, Abdul Salam Aldabik, Amate  
**Project Deadline:** November 24, 2025  
**Presentation:** November 28, 2025
