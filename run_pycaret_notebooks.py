"""
Execute PyCaret notebooks to train models
"""
import subprocess
import sys
import os

def run_notebook(notebook_path):
    """Execute a Jupyter notebook"""
    print(f"\n{'='*80}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*80}\n")
    
    try:
        # Use jupyter nbconvert to execute
        cmd = [
            sys.executable, '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            notebook_path,
            '--output', notebook_path.replace('.ipynb', '_executed.ipynb'),
            '--ExecutePreprocessor.timeout=600'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully executed: {notebook_path}")
            return True
        else:
            print(f"❌ Error executing: {notebook_path}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    # Change to housing code directory
    housing_code = r"C:\Users\abdul\Desktop\ML\AWS\Machine-Learning-Project-TM-2025\Dataset_1_UK_Housing\Code"
    
    notebooks = [
        os.path.join(housing_code, "07_using_PyCaret.ipynb"),
        os.path.join(housing_code, "08_AWS_using_PyCaret.ipynb"),
    ]
    
    results = []
    for notebook in notebooks:
        if os.path.exists(notebook):
            success = run_notebook(notebook)
            results.append((notebook, success))
        else:
            print(f"⚠️  Notebook not found: {notebook}")
            results.append((notebook, False))
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for notebook, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {os.path.basename(notebook)}")
