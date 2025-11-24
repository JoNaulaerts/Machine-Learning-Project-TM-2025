# Oracle Cloud Manual Deployment Script
# Your Oracle VM IP: 158.178.211.8

$ORACLE_HOST = "158.178.211.8"
$ORACLE_USER = "opc"  # Change to "opc" if using Oracle Linux
$SSH_KEY = "path\to\your\ssh\key"  # UPDATE THIS PATH

Write-Host "🚀 Deploying to Oracle Cloud VM: $ORACLE_HOST" -ForegroundColor Green
Write-Host ""

# Step 1: Test SSH Connection
Write-Host "Step 1: Testing SSH connection..." -ForegroundColor Yellow
ssh -i $SSH_KEY ${ORACLE_USER}@${ORACLE_HOST} "echo 'SSH connection successful!'"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ SSH connection failed. Please check:" -ForegroundColor Red
    Write-Host "  - SSH key path is correct" -ForegroundColor Red
    Write-Host "  - Username is correct (ubuntu or opc)" -ForegroundColor Red
    Write-Host "  - VM is running in Oracle Console" -ForegroundColor Red
    exit 1
}

Write-Host "✅ SSH connection successful!" -ForegroundColor Green
Write-Host ""

# Step 2: Setup VM (firewall, directories, Docker)
Write-Host "Step 2: Setting up Oracle VM..." -ForegroundColor Yellow
ssh -i $SSH_KEY ${ORACLE_USER}@${ORACLE_HOST} @'
# Open firewall ports
echo "Opening firewall ports 8501 and 8502..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8502 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || echo "netfilter-persistent not installed, using iptables-save"
sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null 2>&1 || true

# For Oracle Linux alternative
sudo firewall-cmd --permanent --add-port=8501/tcp 2>/dev/null || true
sudo firewall-cmd --permanent --add-port=8502/tcp 2>/dev/null || true
sudo firewall-cmd --reload 2>/dev/null || true

echo "✅ Firewall configured"

# Create directories
echo "Creating project directories..."
mkdir -p ~/ml-project/housing_app
mkdir -p ~/ml-project/electricity_app
mkdir -p ~/ml-project/data/housing
mkdir -p ~/ml-project/data/electricity
echo "✅ Directories created"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install docker-compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing docker-compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ docker-compose installed"
else
    echo "✅ docker-compose already installed"
fi

echo ""
echo "✅ VM setup complete!"
'@

Write-Host "✅ Oracle VM configured!" -ForegroundColor Green
Write-Host ""

# Step 3: Clone repository
Write-Host "Step 3: Cloning GitHub repository..." -ForegroundColor Yellow
ssh -i $SSH_KEY ${ORACLE_USER}@${ORACLE_HOST} @'
cd ~/ml-project
if [ -d "Machine-Learning-Project-TM-2025" ]; then
    echo "Repository exists, pulling latest changes..."
    cd Machine-Learning-Project-TM-2025
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
    cd Machine-Learning-Project-TM-2025
fi
echo "✅ Repository ready"
'@

Write-Host "✅ Repository cloned!" -ForegroundColor Green
Write-Host ""

# Step 4: Upload data files
Write-Host "Step 4: Uploading data files (this may take a while)..." -ForegroundColor Yellow
Write-Host "  Uploading housing data..." -ForegroundColor Cyan

# Check if files exist before uploading
$housingFiles = @(
    "Dataset_1_UK_Housing\Data\price_paid_records.csv",
    "Dataset_1_UK_Housing\Data\economic_indicators_combined.csv"
)

foreach ($file in $housingFiles) {
    if (Test-Path $file) {
        Write-Host "    Uploading $file..." -ForegroundColor Gray
        scp -i $SSH_KEY $file ${ORACLE_USER}@${ORACLE_HOST}:~/ml-project/data/housing/
    } else {
        Write-Host "    ⚠️  File not found: $file" -ForegroundColor Yellow
    }
}

# Upload housing parquet file if exists
$parquetFile = "Dataset_1_UK_Housing\Data\feature_analysis\housing_features_final.parquet"
if (Test-Path $parquetFile) {
    Write-Host "    Uploading $parquetFile..." -ForegroundColor Gray
    scp -i $SSH_KEY $parquetFile ${ORACLE_USER}@${ORACLE_HOST}:~/ml-project/data/housing/
} else {
    Write-Host "    ℹ️  Parquet file not found (optional)" -ForegroundColor Cyan
}

Write-Host "  Uploading electricity data..." -ForegroundColor Cyan
$electricityFile = "Dataset_2_UK_Historic_Electricity_Demand_Data\Data\neso_historic_demand_combined.csv"
if (Test-Path $electricityFile) {
    Write-Host "    Uploading $electricityFile..." -ForegroundColor Gray
    scp -i $SSH_KEY $electricityFile ${ORACLE_USER}@${ORACLE_HOST}:~/ml-project/data/electricity/
} else {
    Write-Host "    ⚠️  File not found: $electricityFile" -ForegroundColor Yellow
}

Write-Host "✅ Data files uploaded!" -ForegroundColor Green
Write-Host ""

# Step 5: Deploy with Docker
Write-Host "Step 5: Building and starting Docker containers..." -ForegroundColor Yellow
ssh -i $SSH_KEY ${ORACLE_USER}@${ORACLE_HOST} @'
cd ~/ml-project/Machine-Learning-Project-TM-2025

# Update docker-compose.yml to use correct data paths
cat > docker-compose.yml << 'COMPOSE_EOF'
version: '3.8'

services:
  housing:
    build:
      context: .
      dockerfile: Dockerfile.housing
    ports:
      - "8501:8501"
    volumes:
      - ./Dataset_1_UK_Housing/Code/streamlit_app.py:/app/streamlit_app.py
      - ../data/housing:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    container_name: housing-app

  electricity:
    build:
      context: .
      dockerfile: Dockerfile.electricity
    ports:
      - "8502:8502"
    volumes:
      - ./Dataset_2_UK_Historic_Electricity_Demand_Data/Code/streamlit_app.py:/app/streamlit_app.py
      - ../data/electricity:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8502
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    container_name: electricity-app
COMPOSE_EOF

echo "Building Docker images..."
docker-compose build

echo "Starting containers..."
docker-compose up -d

echo ""
echo "Waiting for containers to start..."
sleep 10

echo ""
echo "Container status:"
docker-compose ps

echo ""
echo "✅ Deployment complete!"
'@

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "🎉 DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host ""
Write-Host "Your applications are now live at:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🏠 Housing Prediction:   http://$ORACLE_HOST:8501" -ForegroundColor Yellow
Write-Host "  ⚡ Electricity Forecast: http://$ORACLE_HOST:8502" -ForegroundColor Yellow
Write-Host ""
Write-Host "To check logs:" -ForegroundColor Gray
Write-Host "  ssh -i $SSH_KEY ${ORACLE_USER}@${ORACLE_HOST}" -ForegroundColor Gray
Write-Host "  cd ~/ml-project/Machine-Learning-Project-TM-2025" -ForegroundColor Gray
Write-Host "  docker-compose logs -f" -ForegroundColor Gray
Write-Host ""
