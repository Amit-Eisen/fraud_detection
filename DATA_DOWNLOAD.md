# 📊 Dataset Download Instructions

## Dataset Information

**Name**: Credit Card Fraud Detection Dataset  
**Size**: ~10GB (35,180,112 transactions)  
**Format**: CSV  
**Features**: 23 columns  
**Target**: `is_fraud` (0.61% fraud rate)  

---

## Download Options

### Option 1: Kaggle (Recommended)

**Original Dataset**: [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

**Steps:**
1. Go to Kaggle dataset page (link above)
2. Click "Download" button (requires Kaggle account - free!)
3. Extract the downloaded ZIP file
4. Rename to `credit_card_fraud.csv`
5. Place in: `fraud_detection/data/raw/`

**Using Kaggle API (Advanced):**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (get from kaggle.com/account)
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d nelgiriyewithana/credit-card-fraud-detection-dataset-2023

# Extract and move
unzip credit-card-fraud-detection-dataset-2023.zip
mv creditcard_2023.csv fraud_detection/data/raw/credit_card_fraud.csv
```

---

### Option 2: Google Drive (If you host it)

If you want to share your processed version:

1. Upload to Google Drive
2. Set sharing to "Anyone with the link"
3. Get shareable link
4. Use `gdown` to download:

```bash
pip install gdown

# Download from Google Drive
gdown https://drive.google.com/uc?id=YOUR_FILE_ID

# Move to project
mv credit_card_fraud.csv fraud_detection/data/raw/
```

---

### Option 3: Direct Download (If hosted elsewhere)

```bash
# Using wget
cd fraud_detection/data/raw/
wget https://your-hosting-url.com/credit_card_fraud.csv

# OR using curl
curl -O https://your-hosting-url.com/credit_card_fraud.csv
```

---

## After Download

### Verify the file:
```bash
cd fraud_detection/data/raw/
ls -lh credit_card_fraud.csv
# Should show ~10GB

# Check first few lines
head -5 credit_card_fraud.csv
```

Expected columns:
```
id,trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long,is_fraud
```

### Convert to Parquet (Recommended):
```bash
# Activate virtual environment
source .venv/bin/activate

# Convert CSV to Parquet (much faster for processing!)
python src/utils/convert_to_parquet.py
```

This will create:
- `data/raw/credit_card_fraud.parquet` (~3GB, 3x faster to load!)

---

## Dataset Statistics

- **Total Transactions**: 35,180,112
- **Fraud Cases**: 214,597 (0.61%)
- **Legitimate Cases**: 34,965,515 (99.39%)
- **Date Range**: 2019-2020
- **Features**: 
  - Transaction: amount, category, merchant
  - Customer: age, gender, location, job
  - Temporal: date, time, day of week
  - Geospatial: lat, long, city population

---

## Troubleshooting

**Problem**: Download is too slow  
**Solution**: Use Kaggle CLI with resume capability

**Problem**: Not enough disk space  
**Solution**: You need ~15GB free (10GB CSV + 3GB Parquet + 2GB working space)

**Problem**: File is corrupted  
**Solution**: Re-download and verify file size (~10GB)

**Problem**: Can't find the file after download  
**Solution**: Make sure it's in `fraud_detection/data/raw/credit_card_fraud.csv`

---

## Alternative: Use Smaller Sample

If you have limited resources, you can work with a sample:

```python
import pandas as pd

# Load and sample
df = pd.read_csv('data/raw/credit_card_fraud.csv', nrows=1_000_000)  # 1M rows
df.to_csv('data/raw/credit_card_fraud_sample.csv', index=False)
```

**Note**: The full project was designed for the complete 35M dataset!

---

## Security Note

⚠️ **NEVER commit the raw data to Git!**

The `.gitignore` file already excludes:
- `data/raw/*.csv`
- `data/raw/*.parquet`

This prevents accidentally uploading 10GB to GitHub.

---

## Questions?

If you have issues downloading the dataset:
1. Check the Kaggle dataset page for updates
2. Ensure you have enough disk space
3. Try the Kaggle CLI method
4. Contact the dataset author on Kaggle

---

**Ready to start?** Once downloaded, proceed to `SETUP.md` for the full pipeline!

