# AWS SageMaker Training Instructions

**For Teammate:** Complete instructions to run AWS SageMaker models and return results

**Date:** November 24, 2025  
**Project:** Machine-Learning-Project-TM-2025

---

## 📋 Overview

You need to run 2 notebooks on AWS SageMaker:
1. `09_AWS_SageMaker_Housing_LinearLearner.ipynb` - Housing Price Prediction (Linear Learner)
2. `07_AWS_SageMaker_Electricity_DeepAR.ipynb` - Electricity Demand Forecasting (DeepAR)

**Total time:** ~30-45 minutes  
**Est. cost:** ~$5-10 USD

---

## 🚀 Step-by-Step Instructions

### PART 1: Setup AWS SageMaker

#### 1.1 Access AWS SageMaker Console
1. Log into AWS Console: https://console.aws.amazon.com/
2. Search for "SageMaker" in services
3. Click on "Amazon SageMaker"

#### 1.2 Create Notebook Instance
1. In SageMaker console, click **"Notebook instances"** (left sidebar)
2. Click **"Create notebook instance"** button
3. Configure:
   - **Name:** `ml-project-notebook` (or any name)
   - **Instance type:** `ml.t3.medium` or `ml.t3.large`
   - **Platform:** JupyterLab 3
   - **IAM Role:** Create new role
     - Select "Any S3 bucket"
     - Click "Create role"
4. Leave other settings as default
5. Click **"Create notebook instance"**
6. **Wait 3-5 minutes** for status to change from "Pending" to "InService"

---

### PART 2: Upload Files

#### 2.1 Open JupyterLab
1. Once status is "InService", click **"Open JupyterLab"**
2. JupyterLab interface will open in new tab

#### 2.2 Upload Notebooks
1. In JupyterLab, click the **Upload** button (↑ icon)
2. Upload these files:
   - `09_AWS_SageMaker_Housing_LinearLearner.ipynb` (from `Dataset_1_UK_Housing/Code/`)
   - `07_AWS_SageMaker_Electricity_DeepAR.ipynb` (from `Dataset_2_UK_Historic_Electricity_Demand_Data/Code/`)

#### 2.3 Upload Data Files
1. Click Upload again
2. Upload:
   - `housing_features_final.parquet` (from `Dataset_1_UK_Housing/Data/`)
   - `neso_historic_demand_combined.csv` (from `Dataset_2_UK_Historic_Electricity_Demand_Data/Data/`)

**Note:** If files are too large (>100MB), you can:
- Upload to S3 bucket first
- Modify notebooks to download from S3 (code provided in notebooks)

---

### PART 3: Run Housing Model (Notebook 1)

#### 3.1 Open Housing Notebook
1. Double-click `09_AWS_SageMaker_Housing_LinearLearner.ipynb`
2. Select kernel: **conda_python3** (if prompted)

#### 3.2 Fix Data Loading (IMPORTANT)
1. Find **Cell "3. Load and Prepare Data"**
2. Replace the placeholder code with:
   ```python
   # Load from uploaded file
   df = pd.read_parquet('housing_features_final.parquet')
   print(f"Dataset shape: {df.shape}")
   print(f"\nColumns: {df.columns.tolist()}")
   df.head()
   ```

#### 3.3 Run All Cells
1. Click **"Run" menu → "Run All Cells"**
2. **Or** manually run each cell with Shift+Enter

#### 3.4 Monitor Progress
Watch for these stages (approx. times):
- ✅ Setup & Data Loading: 1-2 minutes
- ✅ Upload to S3: 2-3 minutes
- ⏱️ **Training Job:** 5-10 minutes (shows progress logs)
- ⏱️ **Deployment:** 5-10 minutes
- ✅ Predictions & Evaluation: 2-3 minutes

**Total: ~20-30 minutes**

#### 3.5 Important Notes
- Training will show logs like `Training image download`, `Starting training job...`
- Deployment shows `Creating endpoint...`
- **DO NOT CLOSE** the browser during training/deployment

#### 3.6 Results
At the end, you'll see:
```
AWS SageMaker Linear Learner - Model Performance
==========================================================
Root Mean Squared Error (RMSE): [VALUE]
Mean Absolute Error (MAE):     [VALUE]
R² Score:                      [VALUE]
```

**Copy these numbers!** You'll need them.

---

### PART 4: Run Electricity Model (Notebook 2)

#### 4.1 Open Electricity Notebook
1. Double-click `07_AWS_SageMaker_Electricity_DeepAR.ipynb`
2. Select kernel: **conda_python3**

#### 4.2 Fix Data Loading
1. Find **Cell "3. Load and Prepare Data"**
2. Replace with:
   ```python
   # Load from uploaded file
   df = pd.read_csv('neso_historic_demand_combined.csv')
   df['SETTLEMENT_DATE'] = pd.to_datetime(df['SETTLEMENT_DATE'])
   df = df.sort_values(['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'])
   print(f"Dataset shape: {df.shape}")
   print(f"Date range: {df['SETTLEMENT_DATE'].min()} to {df['SETTLEMENT_DATE'].max()}")
   df.head()
   ```

#### 4.3 Run All Cells
1. Click **"Run" → "Run All Cells"**

#### 4.4 Monitor Progress
Stages:
- ✅ Setup & Data Prep: 2-3 minutes
- ✅ JSON formatting: 1-2 minutes
- ⏱️ **DeepAR Training:** 10-20 minutes
- ⏱️ **Deployment:** 5-10 minutes
- ✅ Forecasting & Evaluation: 2-3 minutes

**Total: ~25-40 minutes**

#### 4.5 Results
You'll see:
```
AWS SageMaker DeepAR - Model Performance
==========================================================
Root Mean Squared Error (RMSE): [VALUE] MW
Mean Absolute Error (MAE):     [VALUE] MW
Mean Absolute % Error (MAPE):  [VALUE]%
R² Score:                      [VALUE]
```

**Copy these numbers!**

---

### PART 5: Download Results

#### 5.1 Download Executed Notebooks
1. In JupyterLab file browser, **right-click** on each notebook
2. Select **"Download"**
3. Download:
   - `09_AWS_SageMaker_Housing_LinearLearner.ipynb` (with outputs)
   - `07_AWS_SageMaker_Electricity_DeepAR.ipynb` (with outputs)

#### 5.2 Download Result JSON Files
1. Find in file browser:
   - `aws_sagemaker_results.json`
   - `aws_deepar_results.json`
2. Right-click each → Download

#### 5.3 Download Any Generated Plots
1. Look for any .png files created
2. Download them too

---

### PART 6: Clean Up (CRITICAL!)

#### 6.1 Delete Endpoints (Stops Hourly Charges!)
**The notebooks should do this automatically in the last cells, but verify:**

1. In SageMaker console, go to **"Inference" → "Endpoints"**
2. Check if any endpoints are listed
3. If yes:
   - Select the endpoint
   - Click **"Delete"**
   - Confirm deletion

#### 6.2 Stop Notebook Instance
1. Go to **"Notebook instances"**
2. Select your instance (`ml-project-notebook`)
3. Click **"Actions" → "Stop"**
4. Wait for status to change to "Stopped"

**Or permanently delete:**
- Click **"Actions" → "Delete"**
- Confirm deletion

#### 6.3 Verify No Charges
1. Go to **"AWS Billing Dashboard"**
2. Check for any running resources
3. Estimated charges should stop accumulating

---

## 📦 What to Send Back to Abdul

### Required Files:
1. ✅ `09_AWS_SageMaker_Housing_LinearLearner.ipynb` (executed with outputs)
2. ✅ `07_AWS_SageMaker_Electricity_DeepAR.ipynb` (executed with outputs)
3. ✅ `aws_sagemaker_results.json`
4. ✅ `aws_deepar_results.json`
5. ✅ Any generated plots/images

### Required Information:
Copy and paste these results:

**Housing Model Results:**
```
RMSE: _______
MAE: _______
R² Score: _______
Training Time: _______ minutes
```

**Electricity Model Results:**
```
RMSE: _______ MW
MAE: _______ MW
MAPE: _______% 
R² Score: _______
Training Time: _______ minutes
```

---

## 🆘 Troubleshooting

### Problem: "No module named 'sagemaker'"
**Solution:** Run this in a code cell:
```python
!pip install sagemaker boto3 pandas scikit-learn
```

### Problem: "Access Denied" or "Role error"
**Solution:** When creating notebook instance, make sure you created a new IAM role with S3 access.

### Problem: "Training job failed"
**Solution:** 
1. Check training logs in cell output
2. Common issues:
   - Data format incorrect (CSV should have no header for Linear Learner)
   - S3 permissions issue (recreate IAM role)
   - Verify data uploaded correctly

### Problem: "Endpoint creation failed"
**Solution:**
1. Check you have service limits for ml.m4.xlarge
2. Try smaller instance type: `ml.t2.medium`

### Problem: Files too large to upload
**Solution:**
1. Upload files to S3 bucket first:
   - Go to S3 console
   - Create bucket or use existing
   - Upload files
2. In notebook, download from S3:
   ```python
   import boto3
   s3 = boto3.client('s3')
   s3.download_file('your-bucket', 'housing_features_final.parquet', 'housing_features_final.parquet')
   ```

### Problem: "Kernel died" or notebook frozen
**Solution:**
1. Kernel → Restart Kernel
2. Run cells again from where it stopped

---

## 💰 Cost Breakdown

**Estimated Costs:**
- Notebook instance (ml.t3.medium): ~$0.05/hour
- Training instance (ml.m4.xlarge): ~$0.23/hour
- Endpoint instance (ml.m4.xlarge): ~$0.23/hour
- S3 storage: ~$0.01

**Total for this task:** $5-10 USD (if you delete endpoints immediately)

**⚠️ WARNING:** Leaving endpoints running costs $5.52/day!

---

## ✅ Final Checklist

Before sending files back:

- [ ] Both notebooks executed completely (all cells run)
- [ ] Model performance metrics visible in notebook outputs
- [ ] Plots/visualizations generated
- [ ] JSON result files downloaded
- [ ] Endpoints deleted (check SageMaker console)
- [ ] Notebook instance stopped or deleted
- [ ] No unexpected AWS charges appearing

---

## 📧 Questions?

If you encounter issues:
1. Take screenshot of error
2. Note which cell number failed
3. Copy error message
4. Send to Abdul with details

---

**Good luck! This should take about 45 minutes total. Remember to delete endpoints when done! 🚀**
