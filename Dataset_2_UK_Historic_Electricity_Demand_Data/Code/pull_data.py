import requests
import pandas as pd
import os
from pathlib import Path
import time

# Create directory for downloads
download_dir = Path("Data")
download_dir.mkdir(exist_ok=True)

print("Fetching dataset metadata from NESO API...")

# Get all resources from the API
api_url = "https://api.neso.energy/api/3/action/datapackage_show?id=historic-demand-data"
response = requests.get(api_url)
data = response.json()

# Extract all CSV resources
csv_resources = []
if data['success']:
    resources = data['result']['resources']
    for resource in resources:
        if resource['format'].upper() == 'CSV':
            csv_resources.append({
                'name': resource['name'],
                'url': resource['url'],
                'filename': resource['url'].split('/')[-1]
            })

print(f"Found {len(csv_resources)} CSV files to download\n")

# Download all CSV files
downloaded_files = []
for i, resource in enumerate(csv_resources, 1):
    print(f"[{i}/{len(csv_resources)}] Downloading {resource['name']}...")
    
    try:
        response = requests.get(resource['url'], timeout=60)
        response.raise_for_status()
        
        filepath = download_dir / resource['filename']
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        downloaded_files.append(filepath)
        print(f"  ✓ Saved to {filepath} ({len(response.content) / 1024 / 1024:.2f} MB)")
        
        # Small delay to be respectful to the server
        time.sleep(0.5)
        
    except Exception as e:
        print(f"  ✗ Error downloading {resource['name']}: {e}")

print(f"\n{'='*60}")
print(f"Downloaded {len(downloaded_files)} files successfully")
print(f"{'='*60}\n")

# Combine all CSV files
print("Combining all CSV files into one dataset...")

all_dataframes = []
for filepath in downloaded_files:
    try:
        print(f"Reading {filepath.name}...")
        df = pd.read_csv(filepath)
        all_dataframes.append(df)
        print(f"  ✓ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"  ✗ Error reading {filepath.name}: {e}")

if all_dataframes:
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by date if SETTLEMENT_DATE column exists
    if 'SETTLEMENT_DATE' in combined_df.columns:
        combined_df['SETTLEMENT_DATE'] = pd.to_datetime(combined_df['SETTLEMENT_DATE'], format='%Y%m%d', errors='coerce')
        combined_df = combined_df.sort_values(['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'], ignore_index=True)
    
    # Save combined dataset
    output_file = "neso_historic_demand_combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Combined dataset saved to: {output_file}")
    print(f"  Total rows: {len(combined_df):,}")
    print(f"  Total columns: {len(combined_df.columns)}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")
    
    # Display first few rows
    print("\nPreview of combined data:")
    print(combined_df.head())
    
    # Display basic statistics
    if 'SETTLEMENT_DATE' in combined_df.columns:
        print(f"\nDate range: {combined_df['SETTLEMENT_DATE'].min()} to {combined_df['SETTLEMENT_DATE'].max()}")
else:
    print("No data to combine!")
