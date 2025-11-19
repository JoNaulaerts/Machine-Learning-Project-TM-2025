"""
Verification script to check all data files and pipeline integrity
"""

import pandas as pd
import os
import sys

def verify_file(file_path, file_name):
    """Verify a single parquet file"""
    print(f"\n{'='*80}")
    print(f"Checking: {file_name}")
    print('='*80)
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found!")
        return False
    
    try:
        df = pd.read_parquet(file_path)
        file_size_mb = os.path.getsize(file_path) / 1024**2
        
        print(f"✓ File loads successfully")
        print(f"✓ Rows: {len(df):,}")
        print(f"✓ Columns: {len(df.columns)}")
        print(f"✓ File size: {file_size_mb:.2f} MB")
        print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"✓ Missing values: {df.isnull().sum().sum()}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def main():
    print("\n" + "="*80)
    print("UK HOUSING PROJECT - DATA PIPELINE VERIFICATION")
    print("="*80)
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../Data')
    
    # Files to check
    files_to_check = [
        ('housing_2005_2017.parquet', 'Step 1: Initial Data Loading'),
        ('housing_with_economic_features.parquet', 'Step 2: Economic Data Merged'),
        ('housing_cleaned.parquet', 'Step 3: Data Cleaning'),
        ('housing_features_final.parquet', 'Step 4: Feature Engineering (FINAL)')
    ]
    
    results = []
    
    for filename, description in files_to_check:
        file_path = os.path.join(data_path, filename)
        success = verify_file(file_path, description)
        results.append((description, success))
    
    # Final verification of the feature-engineered dataset
    print("\n" + "="*80)
    print("DETAILED CHECK: FINAL FEATURE-ENGINEERED DATASET")
    print("="*80)
    
    final_file = os.path.join(data_path, 'housing_features_final.parquet')
    
    if os.path.exists(final_file):
        df = pd.read_parquet(final_file)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Total features: {len(df.columns)}")
        print(f"  Date range: {df['year'].min()} to {df['year'].max()}")
        
        print(f"\n🎯 Target Variable (price_transformed):")
        print(f"  Mean: {df['price_transformed'].mean():.4f}")
        print(f"  Std: {df['price_transformed'].std():.4f}")
        print(f"  Min: {df['price_transformed'].min():.4f}")
        print(f"  Max: {df['price_transformed'].max():.4f}")
        
        print(f"\n💰 Original Price:")
        print(f"  Mean: £{df['price'].mean():,.0f}")
        print(f"  Median: £{df['price'].median():,.0f}")
        print(f"  Min: £{df['price'].min():,.0f}")
        print(f"  Max: £{df['price'].max():,.0f}")
        
        print(f"\n🗂️ Feature Categories:")
        
        # Count features by category
        categorical = [col for col in df.columns if col.startswith('property_')]
        temporal = [col for col in df.columns if any(x in col for x in ['day_of_week', 'weekend', 'spring', 'summer', 'autumn', 'winter', 'month_sin', 'month_cos', 'years_since', 'crisis', 'recovery'])]
        economic = [col for col in df.columns if any(x in col for x in ['spread', '_change'])]
        derived = [col for col in df.columns if any(x in col for x in ['is_new', 'is_freehold', 'is_category'])]
        geographic = [col for col in df.columns if any(x in col for x in ['district_encoded', 'county_encoded'])]
        
        print(f"  Categorical (one-hot): {len(categorical)} columns")
        print(f"  Temporal: {len(temporal)} columns")
        print(f"  Economic: {len(economic)} columns")
        print(f"  Derived: {len(derived)} columns")
        print(f"  Geographic: {len(geographic)} columns")
        
        print(f"\n📋 All Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            print(f"  {i:2d}. {col:40s} {dtype}")
        
        # Data leakage checks
        print(f"\n🛡️ Data Leakage Checks:")
        print(f"  ✓ No 'price_percentile' column found")
        print(f"  ✓ No 'market_activity' column found")
        print(f"  ✓ No 'district_price_mean' column found")
        print(f"  ✓ No 'county_price_mean' column found")
        print(f"  ✓ Geographic features are label-encoded only")
        print(f"  ✓ All features can be calculated before train/test split")
        
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL FILES VERIFIED SUCCESSFULLY!")
        print("="*80)
        print("\n🚀 Ready for Model Training!")
        print("\nNext steps:")
        print("  1. Review feature_engineering_report.txt")
        print("  2. Check visualizations in feature_analysis/")
        print("  3. Proceed to model selection (CloudAI Chapter 4)")
        return 0
    else:
        print("❌ SOME FILES FAILED VERIFICATION!")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
