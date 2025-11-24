# Dataset 2: UK Historic Electricity Demand Data

## Overview
Forecasting electricity demand in England/Wales using 2001-2025 half-hourly data from NESO (National Energy System Operator).

**Worked on by:** Abdul Salam Aldabik

---

## 📁 Project Structure

### Code/ (Numbered Execution Order)

| File | Description | Status |
|------|-------------|--------|
| **00_final_data_preparation.ipynb** | Clean data pipeline (no exploration) | ✅ Complete |
| **02_data_pull.ipynb** | Download & combine CSVs from NESO | ✅ Complete |
| **03_data_cleaning_and_eda.ipynb** | Full EDA with graphs, outlier analysis | ✅ Complete |
| **04_time_series_analysis.ipynb** | Time series decomposition & patterns | ✅ Complete |
| **05_exploratory_models.ipynb** | Initial model experiments | ✅ Complete |
| **06_complete_model_training.ipynb** | **MAIN: 5 models trained (Prophet/XGBoost/LSTM/Ensemble/PyCaret)** | ✅ Complete |
| **07_final_model_comparison.ipynb** | Compare all models, visualizations, conclusions | ✅ Complete |
| **streamlit_app.py** | Deployment app with forecasting UI | ✅ Complete |

### Data/
- Raw CSVs (demanddata_2001.csv to demanddata_2025.csv)
- `cleaned_and_augmented_electricity_data.csv` (Final dataset)
- Saved models (.pkl, .h5 files)
- Metrics CSVs (complete_model_comparison.csv)

### Output/
- Visualizations (PNGs)
- Model comparison reports

### old_codes/
- Experimental notebooks (not part of final submission)
- AWS SageMaker model (06_aws_sagemaker_model.ipynb)

---

## 🎯 Models Trained

| Model | MAPE | MAE | RMSE | R² | Training Time |
|-------|------|-----|------|-----|---------------|
| **XGBoost** | 3.00% | 751 MW | 1,070 MW | 0.941 | 5.6s |
| **Ensemble** | 4.71% | 1,129 MW | 1,417 MW | 0.897 | <1s |
| **LSTM** | 7.23% | 1,710 MW | 2,430 MW | 0.696 | 927s |
| **Prophet** | 17.77% | 4,072 MW | 4,892 MW | -0.230 | 232s |
| **PyCaret AutoML** | TBD | TBD | TBD | TBD | Auto |

**Winner:** XGBoost (3% MAPE - best accuracy + fastest training)

---

## 🚀 How to Run

### 1. Data Preparation
```bash
jupyter notebook Code/00_final_data_preparation.ipynb
```

### 2. Train All Models
```bash
jupyter notebook Code/06_complete_model_training.ipynb
```

### 3. Compare Models
```bash
jupyter notebook Code/07_final_model_comparison.ipynb
```

### 4. Deploy Streamlit App
```bash
streamlit run Code/streamlit_app.py
```

---

## 📊 Key Features

**Data Processing:**
- ✅ 25 years of data (2001-2025)
- ✅ Half-hourly granularity (48 periods/day)
- ✅ Forward fill for missing values (time series best practice)
- ✅ Outlier capping (0.5th-99.5th percentiles)
- ✅ 9 temporal features (year, month, day, hour, weekend, etc.)

**Models:**
- ✅ Statistical (Prophet with full seasonality)
- ✅ Machine Learning (XGBoost with 24 engineered features)
- ✅ Deep Learning (Bidirectional LSTM)
- ✅ Ensemble (weighted average of all 3)
- ✅ AutoML (PyCaret comparison)
- ✅ AWS SageMaker (cloud deployment)

**Deployment:**
- ✅ Streamlit frontend
- ✅ Model selection UI
- ✅ Interactive forecasting
- ✅ Visualization of predictions

---

## 📈 Results Summary

**Best Performance:** XGBoost
- Predicts within 3% of actual demand
- For 30,000 MW demand → ~900 MW error
- Fastest training (5.6 seconds)
- Best R² score (0.941)

**Most Robust:** Ensemble
- Combines strengths of all models
- 4.71% MAPE
- Recommended for critical applications

---

## 🎓 Presentation Notes

**What we found in EDA:**
- Strong daily seasonality (48 periods)
- Weekly patterns (weekday vs weekend)
- Yearly seasonality (summer vs winter demand)
- Outliers during extreme weather events

**Challenges:**
- Large dataset (900K+ rows)
- Missing values in interconnector flows
- Half-hourly forecasting complexity

**Solutions:**
- Forward fill for time series integrity
- Feature engineering (lag features, rolling stats)
- Multiple model types for comparison

---

## 📝 Files Required for Submission

✅ **EDA:** 03_data_cleaning_and_eda.ipynb (cleaning + graphs + explanations)  
✅ **Final Import:** 00_final_data_preparation.ipynb (clean pipeline)  
✅ **Models:** 06_complete_model_training.ipynb (5 models in one file)  
✅ **Comparison:** 07_final_model_comparison.ipynb (metrics + conclusions)  
✅ **Deployment:** streamlit_app.py (frontend)  
✅ **AWS Model:** old_codes/06_aws_sagemaker_model.ipynb  

---

## ⏭️ Next Steps

1. ✅ Add author attribution to all notebooks
2. ✅ Test Streamlit app with all models
3. ✅ Update main README.md
4. 📝 Prepare presentation slides
5. 🚀 Optional: Deploy to cloud (Oracle/AWS)

---

**Ready for submission:** ✅  
**Presentation date:** 28 November 2025
