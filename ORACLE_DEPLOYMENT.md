# Oracle Cloud Deployment Guide

This guide will help you deploy the ML project to Oracle Cloud using GitHub Actions automation.

## Prerequisites

- Oracle Cloud account (Free Tier)
- VM instance created
- SSH key pair for the VM

## Step 1: Configure Oracle VM

### 1.1 Open Required Ports

In Oracle Cloud Console:

1. Navigate to **Networking** → **Virtual Cloud Networks**
2. Click on your VCN → **Security Lists** → **Default Security List**
3. Click **Add Ingress Rules** and add:

**Rule 1 - Streamlit Housing App:**
- Source CIDR: `0.0.0.0/0`
- IP Protocol: `TCP`
- Destination Port Range: `8501`
- Description: `Streamlit Housing App`

**Rule 2 - Streamlit Electricity App:**
- Source CIDR: `0.0.0.0/0`
- IP Protocol: `TCP`
- Destination Port Range: `8502`
- Description: `Streamlit Electricity App`

### 1.2 Configure VM Firewall

SSH into your Oracle VM and run:

```bash
# Open ports in iptables
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8502 -j ACCEPT

# Save rules
sudo netfilter-persistent save

# Or for Oracle Linux:
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --permanent --add-port=8502/tcp
sudo firewall-cmd --reload
```

### 1.3 Create Project Directory

```bash
mkdir -p ~/ml-project/housing_app
mkdir -p ~/ml-project/electricity_app
mkdir -p ~/ml-project/data/housing
mkdir -p ~/ml-project/data/electricity
```

## Step 2: Upload Data Files to Oracle VM

The large data files are gitignored and need to be uploaded separately:

```powershell
# From your local machine (PowerShell)

# Replace with your Oracle VM details
$ORACLE_HOST = "YOUR_VM_PUBLIC_IP"
$ORACLE_USER = "ubuntu"  # or "opc" for Oracle Linux
$SSH_KEY = "path\to\your\private\key"

# Upload Housing data
scp -i $SSH_KEY `
  Dataset_1_UK_Housing\Data\price_paid_records.csv `
  Dataset_1_UK_Housing\Data\economic_indicators_combined.csv `
  Dataset_1_UK_Housing\Data\feature_analysis\housing_features_final.parquet `
  ${ORACLE_USER}@${ORACLE_HOST}:~/ml-project/data/housing/

# Upload Electricity data
scp -i $SSH_KEY `
  Dataset_2_UK_Historic_Electricity_Demand_Data\Data\neso_historic_demand_combined.csv `
  ${ORACLE_USER}@${ORACLE_HOST}:~/ml-project/data/electricity/
```

## Step 3: Configure GitHub Secrets

Go to your GitHub repository:

1. Navigate to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add:

### Required Secrets:

**ORACLE_HOST**
- Value: Your VM's public IP address
- Example: `123.45.67.89`

**ORACLE_USER**
- Value: Your SSH username
- Ubuntu: `ubuntu`
- Oracle Linux: `opc`

**ORACLE_SSH_KEY**
- Value: Your private SSH key content
- To get the key content:
  ```powershell
  Get-Content path\to\your\private\key | Out-String
  ```
- Copy the entire output including:
  ```
  -----BEGIN RSA PRIVATE KEY-----
  [key content]
  -----END RSA PRIVATE KEY-----
  ```

## Step 4: Deploy via GitHub Actions

Once secrets are configured:

```bash
git add .
git commit -m "Add Oracle Cloud automated deployment"
git push origin main
```

GitHub Actions will automatically:
1. ✅ Train models
2. ✅ Build Docker images
3. ✅ Deploy to Oracle Cloud
4. ✅ Start Streamlit apps

## Step 5: Access Your Applications

After deployment completes (check GitHub Actions):

- **Housing Prediction App:** `http://YOUR_ORACLE_IP:8501`
- **Electricity Forecast App:** `http://YOUR_ORACLE_IP:8502`

## Troubleshooting

### Can't connect to apps?

1. **Check GitHub Actions logs** for deployment errors
2. **Verify ports are open:**
   ```bash
   ssh -i your_key user@oracle_ip
   sudo iptables -L -n | grep 850
   docker-compose ps
   ```

3. **Check container logs:**
   ```bash
   cd ~/ml-project
   docker-compose logs housing
   docker-compose logs electricity
   ```

### Deployment fails?

1. **Verify SSH connection manually:**
   ```bash
   ssh -i your_key user@oracle_ip "echo Connection successful"
   ```

2. **Check data files exist:**
   ```bash
   ssh -i your_key user@oracle_ip "ls -lh ~/ml-project/data/housing/"
   ```

3. **Ensure Docker is running:**
   ```bash
   ssh -i your_key user@oracle_ip "sudo systemctl status docker"
   ```

## Manual Deployment (Alternative)

If GitHub Actions deployment isn't needed, you can deploy manually:

```bash
# SSH into Oracle VM
ssh -i your_key user@oracle_ip

# Clone repository
git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
cd Machine-Learning-Project-TM-2025

# Install Docker (if needed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Start applications
docker-compose up -d

# Check status
docker-compose ps
```

## Cost Considerations

Oracle Free Tier includes:
- ✅ 2 AMD-based Compute VMs (1/8 OCPU, 1 GB RAM each)
- ✅ OR 4 Arm-based Ampere A1 cores + 24 GB RAM total
- ✅ 200 GB Block Volume
- ✅ 10 TB outbound data transfer/month

**This deployment uses minimal resources and stays within free tier limits.**

## Security Notes

- **Never commit SSH keys to GitHub** - Use GitHub Secrets only
- Change default SSH port (22) for additional security
- Consider setting up fail2ban to prevent brute force attacks
- Use Oracle Cloud's built-in DDoS protection

## Monitoring

Monitor your deployment:

```bash
# Watch container resource usage
docker stats

# View real-time logs
docker-compose logs -f

# Check disk usage
df -h
```

---

**For presentation:** Take screenshots of your apps running at `http://YOUR_IP:8501` and `http://YOUR_IP:8502` to demonstrate the deployment! 🎉
