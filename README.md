# Machine-Learning-Project-TM-2025

## Group Name: CloudAI Analytics Team

## Team Members:
- Jo Naulaerts
- Abdul Salam Aldabik

---

## 📋 Project Overview

This project analyzes two datasets using machine learning techniques:

1. **Dataset 1: UK Housing Prices** (1995-2017)
   - Regression models to predict house prices
   - Economic indicators integration
   - Multiple ML approaches (Ridge, PyCaret, AWS SageMaker)

2. **Dataset 2: UK Historic Electricity Demand** (2001-2025)
   - Time series forecasting
   - 4 advanced models: Prophet, XGBoost, LSTM, Ensemble
   - Streamlit deployment for interactive forecasting

---

## 🗂️ Repository Structure

```
Machine-Learning-Project-TM-2025/
│
├── Dataset_1_UK_Housing/
│   ├── Code/
│   │   ├── 00_initial_setup.ipynb
│   │   ├── 01_data_loading.ipynb
│   │   ├── 02_economic_integration.ipynb
│   │   ├── 03_data_merging.ipynb
│   │   ├── 04_data_cleaning.ipynb
│   │   ├── 05_feature_engineering.ipynb
│   │   ├── 06_first_simple_model.ipynb
│   │   ├── 07_using_PyCaret.ipynb
│   │   ├── 08_AWS_using_PyCaret.ipynb
│   │   ├── 09_AWS_SageMaker_Model.ipynb          ⭐ Run in AWS
│   │   ├── 10_final_model_comparison.ipynb
│   │   └── streamlit_app.py                      🚀 Deployment
│   └── Data/
│       └── (Large datasets - not committed to Git)
│
├── Dataset_2_UK_Historic_Electricity_Demand_Data/
│   ├── Code/
│   │   ├── 00_final_data_preparation.ipynb
│   │   ├── 01_data_pull.ipynb
│   │   ├── 02_data_cleaning_and_eda.ipynb
│   │   ├── 03_time_series_analysis.ipynb
│   │   ├── 04_exploratory_models.ipynb
│   │   ├── 05_complete_model_training.ipynb      ⭐ 4 Models
│   │   ├── 06_final_model_comparison.ipynb
│   │   ├── 07_AWS_SageMaker_Model.ipynb          ⭐ Run in AWS
│   │   └── streamlit_app.py                      🚀 Deployment
│   └── Data/
│       └── (CSV files for each year 2001-2025)
│
├── CloudAI/
│   ├── Discussion topics/
│   └── Exercises/
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)
- AWS Account (for SageMaker notebooks)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025.git
   cd Machine-Learning-Project-TM-2025
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset 1: UK Housing Prices

### Workflow

Run notebooks in order:

1. **00-05**: Data loading, cleaning, feature engineering
2. **06**: Simple Ridge regression baseline
3. **07**: PyCaret AutoML model selection
4. **08**: PyCaret training (local)
5. **09**: AWS SageMaker Linear Learner (⚠️ Run in AWS)
6. **10**: Model comparison and conclusions

### Key Features
- Economic indicators (BoE interest rates, GDP, inflation)
- Temporal features (year, quarter, month)
- Location encoding (postcode, county, town)
- Property characteristics (type, tenure, age)

### Models
- **Simple Ridge Regression** - Baseline
- **PyCaret AutoML** - Automated best model selection
- **AWS SageMaker Linear Learner** - Cloud-trained scalable solution

### Deployment

Run the Streamlit app:
```bash
cd Dataset_1_UK_Housing/Code
streamlit run streamlit_app.py
```

---

## ⚡ Dataset 2: UK Electricity Demand

### Workflow

Run notebooks in order:

1. **00**: Final data preparation
2. **01**: Data pulling from NESO API
3. **02**: Cleaning and EDA
4. **03**: Time series analysis
5. **04**: Exploratory models
6. **05**: Complete training (Prophet, XGBoost, LSTM, Ensemble) ⭐
7. **06**: Model comparison
8. **07**: AWS SageMaker DeepAR (⚠️ Run in AWS)

### Models & Performance

| Model | MAPE | RMSE | Best For |
|-------|------|------|----------|
| **XGBoost** | 3% | 872 MW | Production (Best) |
| **Ensemble** | 4% | 1,123 MW | Robust predictions |
| **LSTM** | 7% | 1,845 MW | Pattern learning |
| **Prophet** | 18% | 4,532 MW | Seasonal trends |

### Deployment

Run the Streamlit app:
```bash
cd Dataset_2_UK_Historic_Electricity_Demand_Data/Code
streamlit run streamlit_app.py
```

Features:
- Interactive forecasting (7-365 days)
- Model comparison
- Confidence intervals
- Historical data visualization

---

## ☁️ AWS SageMaker Setup

### Required for Notebooks 09 (Dataset 1) and 07 (Dataset 2)

1. **Access AWS Console** with your student account
2. **Navigate to SageMaker** → Notebooks
3. **Create Notebook Instance:**
   - Name: `HousingModel` or `ElectricityForecast`
   - Instance type: `ml.m4.xlarge`
   - Platform: `notebook-al2-v3`
   - Lifecycle: Select `ml-pipeline`

4. **Upload notebooks** and data files
5. **Run training** (10-20 minutes)
6. **Download results** with outputs
7. **⚠️ Important:** Delete endpoints and stop instance!

---

## 📈 Project Results

### Dataset 1 - Housing Prices
- **Best Model:** (Update after running comparison)
- **R² Score:** (Update with actual)
- **RMSE:** (Update with actual)

### Dataset 2 - Electricity Demand
- **Best Model:** XGBoost
- **MAPE:** 3%
- **RMSE:** 872 MW
- **Forecast Horizon:** Up to 1 year

---

## 🎯 Deliverables

✅ **Complete:**
- All data cleaning and EDA notebooks
- Multiple ML models for both datasets
- AWS SageMaker notebooks (ready to run)
- Model comparison notebooks
- Streamlit deployment apps
- Comprehensive documentation

📅 **Deadline:** November 24, 2025

🎤 **Presentation:** November 28, 2025

---

## 💻 Technology Stack

- **Languages:** Python 3.10
- **ML Libraries:** scikit-learn, XGBoost, Prophet, TensorFlow/Keras
- **AutoML:** PyCaret
- **Cloud:** AWS SageMaker
- **Deployment:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly

---

## 📝 Notes

- Large data files (>100MB) are not committed to Git
- Trained models saved locally in `/Data` folders
- AWS notebooks must be run in SageMaker environment
- Remember to update model comparison notebooks with actual results

---

## 🤝 Contributing

Team members:
1. **Jo Naulaerts** - Dataset 1 contributions, Dataset 2 contributions
2. **Abdul Salam Aldabik** - Dataset 1 contributions, Dataset 2 contribution, Deployment

---

## 📞 Contact

For questions about this project, contact team members via Teams or GitHub issues.

---

**License:** Academic Project - November 2025
