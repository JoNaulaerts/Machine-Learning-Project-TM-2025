# %% [markdown]
# # 01_housing_initial_load.ipynb
# ## UK Housing Price Data - Initial Loading and Exploration
# 
# **Team Members:**
# - [Member 1 Name] - Data loading and subsetting strategy
# - [Member 2 Name] - Initial EDA and data quality checks
# - [Member 3 Name] - Documentation and visualization
# 
# **Dataset:** UK Housing prices from Kaggle (1995-2017, ~2GB)
# 
# **Goals:**
# 1. Load the large CSV file efficiently
# 2. Inspect data structure, types, and quality
# 3. Decide on subsetting strategy (time/location/stratified)
# 4. Save processed subset for further analysis

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("Libraries imported successfully!")
print(f"Pandas version: {pd.__version__}")

# %% [markdown]
# ## Step 1: Initial Data Loading
# 
# Loading a 2GB file requires careful memory management. We'll first load a sample to understand the structure.

# %%
# Define file path
file_path = 'pp-complete.csv'  # Update with your actual file path

# Load first 10000 rows to inspect structure
print("Loading sample data (first 10,000 rows)...")
df_sample = pd.read_csv(file_path, nrows=10000)

print(f"\nSample loaded successfully!")
print(f"Shape: {df_sample.shape}")
print(f"Memory usage: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %%
# Display first few rows
print("First 5 rows of the dataset:")
df_sample.head()

# %%
# Check column names and data types
print("Column Information:")
print(df_sample.info())

# %% [markdown]
# ## Step 2: Clean Column Names
# 
# Following best practices from class materials, we'll standardize column names to snake_case.

# %%
# Store original column names
original_columns = df_sample.columns.tolist()
print("Original columns:", original_columns)

# Clean column names: lowercase, replace spaces with underscores
df_sample.columns = df_sample.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

print("\nCleaned columns:", df_sample.columns.tolist())

# Create a mapping dictionary
column_mapping = dict(zip(original_columns, df_sample.columns))
print("\nColumn mapping:")
for old, new in column_mapping.items():
    print(f"  '{old}' → '{new}'")

# %% [markdown]
# ## Step 3: Data Quality Assessment
# 
# Check for missing values, data types, and basic statistics.

# %%
# Check for missing values
print("Missing values per column:")
missing_df = pd.DataFrame({
    'Column': df_sample.columns,
    'Missing_Count': df_sample.isnull().sum(),
    'Missing_Percentage': (df_sample.isnull().sum() / len(df_sample) * 100).round(2)
})
print(missing_df)

# %%
# Check unique values for categorical columns
print("\nUnique values in categorical columns:")
for col in df_sample.columns:
    unique_count = df_sample[col].nunique()
    if unique_count < 50:  # Likely categorical
        print(f"\n{col}: {unique_count} unique values")
        print(df_sample[col].value_counts().head(10))

# %%
# Statistical summary for numerical columns
print("Statistical Summary for Price:")
print(df_sample['price'].describe())

# Visualize price distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_sample['price'], bins=50, edgecolor='black')
axes[0].set_xlabel('Price (£)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Price Distribution (Sample)')
axes[0].axvline(df_sample['price'].median(), color='red', linestyle='--', label=f'Median: £{df_sample["price"].median():,.0f}')
axes[0].legend()

axes[1].boxplot(df_sample['price'])
axes[1].set_ylabel('Price (£)')
axes[1].set_title('Price Boxplot (Sample)')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4: Date Analysis
# 
# Convert date column to datetime and analyze temporal patterns.

# %%
# Convert date to datetime
df_sample['date_of_transfer'] = pd.to_datetime(df_sample['date_of_transfer'])

# Extract temporal features
df_sample['year'] = df_sample['date_of_transfer'].dt.year
df_sample['month'] = df_sample['date_of_transfer'].dt.month
df_sample['quarter'] = df_sample['date_of_transfer'].dt.quarter

print("Date range in sample:")
print(f"  Earliest: {df_sample['date_of_transfer'].min()}")
print(f"  Latest: {df_sample['date_of_transfer'].max()}")

# %%
# Analyze transactions by year
transactions_by_year = df_sample.groupby('year').size()
print("\nTransactions by year (sample):")
print(transactions_by_year)

# Visualize
plt.figure(figsize=(12, 5))
transactions_by_year.plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Number of Transactions')
plt.title('Transaction Count by Year (Sample Data)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5: Property Type Analysis

# %%
# Analyze property types
print("Property Type Distribution:")
property_counts = df_sample['property_type'].value_counts()
print(property_counts)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].pie(property_counts.values, labels=property_counts.index, autopct='%1.1f%%', startangle=90)
axes[0].set_title('Property Type Distribution')

# Price by property type
df_sample.boxplot(column='price', by='property_type', ax=axes[1])
axes[1].set_xlabel('Property Type')
axes[1].set_ylabel('Price (£)')
axes[1].set_title('Price Distribution by Property Type')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.show()

# %%
# Average price by property type
avg_price_by_type = df_sample.groupby('property_type')['price'].agg(['mean', 'median', 'count'])
print("\nAverage Price by Property Type:")
print(avg_price_by_type)

# %% [markdown]
# ## Step 6: Geographic Analysis

# %%
# Top 10 towns/cities by transaction count
print("Top 10 Towns/Cities by Transaction Count (Sample):")
top_towns = df_sample['town_city'].value_counts().head(10)
print(top_towns)

# Visualize
plt.figure(figsize=(12, 6))
top_towns.plot(kind='barh')
plt.xlabel('Number of Transactions')
plt.ylabel('Town/City')
plt.title('Top 10 Towns/Cities by Transaction Count (Sample)')
plt.tight_layout()
plt.show()

# %%
# Average price by county (top 15)
print("\nAverage Price by County (Top 15):")
avg_price_county = df_sample.groupby('county')['price'].mean().sort_values(ascending=False).head(15)
print(avg_price_county)

plt.figure(figsize=(12, 6))
avg_price_county.plot(kind='barh')
plt.xlabel('Average Price (£)')
plt.ylabel('County')
plt.title('Top 15 Counties by Average Price (Sample)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: Subsetting Strategy Decision
# 
# Given the 2GB size, we need to decide on a subset strategy:
# 
# **Option 1: Time-based (2010-2017)**
# - Pros: Recent data, more relevant for current market analysis
# - Cons: Misses long-term trends
# 
# **Option 2: Location-based (London only or specific regions)**
# - Pros: Focused analysis, manageable size
# - Cons: Not representative of national trends
# 
# **Option 3: Stratified random sample**
# - Pros: Representative across all dimensions, smaller dataset
# - Cons: May miss regional/temporal patterns
# 
# **Option 4: Time + Location combination (e.g., 2010-2017 in major cities)**
# - Pros: Balanced approach, focused but comprehensive
# - Cons: More complex filtering logic

# %%
# Let's estimate the full dataset size
print("Estimating full dataset characteristics...")
total_rows_estimate = 25_000_000  # Approximate based on 2GB file

print(f"\nEstimated total rows: ~{total_rows_estimate:,}")
print(f"Sample size: {len(df_sample):,} rows")
print(f"Sample represents: {len(df_sample)/total_rows_estimate*100:.2f}% of data")

# %% [markdown]
# ## Step 8: Load Full Dataset with Chosen Strategy
# 
# **Team Decision: Time-based subset (2010-2017)**
# 
# We'll load only data from 2010 onwards to focus on recent trends while keeping dataset manageable.

# %%
# Define date filter
start_date = '2010-01-01'
end_date = '2017-12-31'

print(f"Loading data from {start_date} to {end_date}...")
print("This may take several minutes for a 2GB file...")

# Load with date parsing and filtering
# Using chunks to manage memory
chunk_size = 100000
chunks = []

for chunk in pd.read_csv(file_path, chunksize=chunk_size, parse_dates=[2]):
    # Apply column name cleaning
    chunk.columns = chunk.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
    
    # Filter by date
    chunk_filtered = chunk[(chunk['date_of_transfer'] >= start_date) & 
                           (chunk['date_of_transfer'] <= end_date)]
    
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)
    
    print(f"Processed {len(chunks) * chunk_size:,} rows...", end='\r')

# Combine all chunks
df_housing = pd.concat(chunks, ignore_index=True)

print(f"\n\nData loaded successfully!")
print(f"Final shape: {df_housing.shape}")
print(f"Memory usage: {df_housing.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# %%
# Quick verification
print("Date range verification:")
print(f"  Min date: {df_housing['date_of_transfer'].min()}")
print(f"  Max date: {df_housing['date_of_transfer'].max()}")
print(f"\nTotal transactions: {len(df_housing):,}")

# %% [markdown]
# ## Step 9: Save Processed Data
# 
# Save the filtered dataset for faster loading in subsequent notebooks.

# %%
# Save as parquet (more efficient than CSV)
output_path = 'housing_2010_2017.parquet'
df_housing.to_parquet(output_path, compression='gzip')
print(f"Data saved to {output_path}")

# Also save a CSV version if needed
csv_output = 'housing_2010_2017.csv'
df_housing.to_csv(csv_output, index=False)
print(f"CSV version saved to {csv_output}")

# Save summary statistics
summary_file = 'housing_summary.txt'
with open(summary_file, 'w') as f:
    f.write("UK Housing Data Summary (2010-2017)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total Transactions: {len(df_housing):,}\n")
    f.write(f"Date Range: {df_housing['date_of_transfer'].min()} to {df_housing['date_of_transfer'].max()}\n")
    f.write(f"Price Range: £{df_housing['price'].min():,} to £{df_housing['price'].max():,}\n")
    f.write(f"Median Price: £{df_housing['price'].median():,}\n")
    f.write(f"Mean Price: £{df_housing['price'].mean():,.2f}\n")
    f.write(f"\nProperty Types:\n")
    f.write(df_housing['property_type'].value_counts().to_string())

print(f"Summary saved to {summary_file}")

# %% [markdown]
# ## Summary and Next Steps
# 
# ### What we accomplished:
# 1. ✅ Loaded and inspected the large UK Housing dataset
# 2. ✅ Cleaned column names to snake_case format
# 3. ✅ Analyzed data quality and structure
# 4. ✅ Applied time-based filtering (2010-2017) as subsetting strategy
# 5. ✅ Saved processed data in efficient format (Parquet)
# 
# ### Key Findings:
# - Dataset contains property transactions with price, location, and property type
# - Multiple property types: Detached, Semi-detached, Terraced, Flat
# - Geographic coverage across England and Wales
# - Temporal coverage enables trend analysis
# 
# ### Next Steps:
# 1. **Notebook 02**: Deep dive into data cleaning (missing values, outliers)
# 2. **Notebook 03**: Exploratory Data Analysis with visualizations
# 3. **Notebook 04**: Feature engineering for modeling
# 
# ### Team Notes:
# - All team members should review this notebook and understand the filtering logic
# - Discuss if 2010-2017 subset is appropriate or if we need different strategy
# - Next meeting: decide on specific research questions to answer

# %%
print("✅ Notebook completed successfully!")
