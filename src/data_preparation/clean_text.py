# src/data_preparation/clean_text.py

import pandas as pd
import time
from pathlib import Path


def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text columns safely (lowercase, remove special chars) without harming numeric or datetime columns.
    
    This function works on a single DataFrame (or chunk) and can be called repeatedly
    for large datasets processed in chunks.
    """
    # --- Columns to exclude from cleaning ---
    exclude_cols = [
        "trans_date", "trans_time", "unix_time",
        "ssn", "cc_num", "acct_num", "trans_num",
        "lat", "long", "merch_lat", "merch_long",
        "amt", "zip", "is_fraud"
    ]

    # --- Detect text columns ---
    text_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude_cols]

    # --- Clean text columns safely ---
    for col in text_cols:
        if col == "profile":
            # remove ".json" and underscores
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace(".json", "", regex=False)
                .str.replace("_", "", regex=False)
                .str.strip()
            )
        else:
            # general cleaning for normal text columns
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9\s._-]", "", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    # --- Clean SSN specifically (remove dashes only) ---
    if "ssn" in df.columns:
        df["ssn"] = df["ssn"].astype(str).str.replace("-", "", regex=False)

    return df


def run_chunked(input_path: Path, output_path: Path, chunksize: int = 1_000_000):
    """
    Process large CSV files in chunks to avoid memory issues.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        chunksize: Number of rows to process at a time (default: 1 million)
    """
    start = time.time()
    
    print(f"🧹 Starting chunked text cleaning")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process in chunks
    first_chunk = True
    total_rows = 0
    chunk_num = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        rows_in_chunk = len(chunk)
        total_rows += rows_in_chunk
        
        print(f"  Processing chunk {chunk_num}: {rows_in_chunk:,} rows (total: {total_rows:,})")
        
        # Clean the chunk
        chunk = run(chunk)
        
        # Write to CSV (append mode after first chunk)
        if first_chunk:
            chunk.to_csv(output_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(output_path, index=False, mode='a', header=False)
    
    end = time.time()
    
    print(f"\n✅ Text cleaning complete!")
    print(f"   Total rows processed: {total_rows:,}")
    print(f"   Output: {output_path}")
    print(f"⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution (for debugging only)
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    raw_path = data_dir / "raw" / "credit_card_fraud.csv"
    processed_path = data_dir / "processed" / "01_clean_text.csv"
    
    # Use chunked processing for large files
    run_chunked(raw_path, processed_path, chunksize=1_000_000)
