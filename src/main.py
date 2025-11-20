# src/main.py
"""
Fraud Detection Data Pipeline
==============================

Orchestrates the complete data preparation workflow:
1. Load raw data (in chunks to avoid memory issues)
2. Clean text columns
3. Reduce high-cardinality categories
4. Engineer time-based features

Each step saves intermediate results to data/processed/ for debugging.
Designed to handle large datasets (35M+ rows) efficiently.
"""

import time
from pathlib import Path

# Import our data preparation modules
from data_preparation import clean_text, reduce_categories, transform_data


def main(chunksize: int = 1_000_000):
    """
    Execute the complete data preparation pipeline with chunked processing.
    
    Args:
        chunksize: Number of rows to process at a time (default: 1 million)
                   Adjust based on available RAM:
                   - 8GB RAM  → chunksize=500_000
                   - 16GB RAM → chunksize=1_000_000
                   - 32GB RAM → chunksize=2_000_000
    """
    
    print("=" * 70)
    print("FRAUD DETECTION DATA PIPELINE")
    print("=" * 70)
    print(f"💾 Memory-efficient mode: Processing {chunksize:,} rows at a time")
    print("=" * 70)
    print()
    
    pipeline_start = time.time()
    
    # --- Setup paths ---
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    raw_file = raw_dir / "credit_card_fraud.csv"
    clean_file = processed_dir / "01_clean_text.csv"
    reduced_file = processed_dir / "02_reduced_categories.csv"
    final_file = processed_dir / "03_transformed.csv"
    
    # --- Step 0: Check raw data ---
    print("📂 STEP 0: Checking raw data")
    print("-" * 70)
    
    if not raw_file.exists():
        print(f"❌ ERROR: Raw data file not found at {raw_file}")
        print("   Please ensure credit_card_fraud.csv is in the data/raw/ directory.")
        return
    
    # Get file size
    file_size_mb = raw_file.stat().st_size / (1024 ** 2)
    print(f"✅ Found: {raw_file}")
    print(f"   File size: {file_size_mb:,.1f} MB")
    print()
    
    # --- Step 1: Clean text ---
    print("🧹 STEP 1: Text cleaning (chunked)")
    print("-" * 70)
    clean_text.run_chunked(raw_file, clean_file, chunksize=chunksize)
    print(f"💾 Checkpoint saved: {clean_file.name}")
    print()
    
    # --- Step 2: Reduce categories ---
    print("📊 STEP 2: Category reduction (chunked, 2-pass)")
    print("-" * 70)
    reduce_categories.run_chunked(clean_file, reduced_file, top_n=50, chunksize=chunksize)
    print(f"💾 Checkpoint saved: {reduced_file.name}")
    print()
    
    # --- Step 3: Data Transformation ---
    print("⚙️  STEP 3: Data Transformation (chunked)")
    print("-" * 70)
    transform_data.run_chunked(reduced_file, final_file, chunksize=chunksize)
    print(f"💾 Final output saved: {final_file.name}")
    print()
    
    # --- Pipeline summary ---
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start
    
    print("=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nOutput files:")
    print(f"  1. {clean_file.name}")
    print(f"  2. {reduced_file.name}")
    print(f"  3. {final_file.name}")
    print("\n🎯 Ready for EDA and modeling!")
    print()


if __name__ == "__main__":
    import sys
    
    # Allow command-line override of chunksize
    chunksize = 1_000_000  # default
    
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "--chunksize" and len(sys.argv) > 2:
                chunksize = int(sys.argv[2])
                print(f"Using custom chunksize: {chunksize:,}")
        except ValueError:
            print("Invalid chunksize. Using default: 1,000,000")
    
    main(chunksize=chunksize)
