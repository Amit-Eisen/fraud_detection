# src/eda/temporal_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path


def run(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze temporal patterns in fraud transactions.
    
    Key questions:
    - What hours do fraudsters operate? (Night vs day)
    - Which days have more fraud? (Weekdays vs weekends)
    - Are there monthly patterns?
    
    Args:
        df: Input DataFrame with time features and 'is_fraud'
        output_dir: Directory to save reports and plots
    
    Returns:
        Same DataFrame
    """
    start = time.time()
    
    print(f"⏰ Starting temporal analysis")
    print(f"   Dataset: {df.shape[0]:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required columns
    required_cols = ['is_fraud', 'trans_hour']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: Required columns {required_cols} not found!")
        return df
    
    # --- 1. Fraud by Hour ---
    hourly_fraud = df.groupby('trans_hour').agg({
        'is_fraud': ['sum', 'count', 'mean']
    }).reset_index()
    hourly_fraud.columns = ['hour', 'fraud_count', 'total_count', 'fraud_rate']
    hourly_fraud['fraud_rate'] = (hourly_fraud['fraud_rate'] * 100).round(3)
    
    hourly_fraud.to_csv(output_dir / '04_fraud_by_hour.csv', index=False)
    
    # --- 2. Fraud by Weekday (if available) ---
    weekday_fraud = None
    if 'trans_weekday' in df.columns:
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_fraud = df.groupby('trans_weekday').agg({
            'is_fraud': ['sum', 'count', 'mean']
        }).reset_index()
        weekday_fraud.columns = ['weekday', 'fraud_count', 'total_count', 'fraud_rate']
        weekday_fraud['fraud_rate'] = (weekday_fraud['fraud_rate'] * 100).round(3)
        
        # Sort by day of week
        weekday_fraud['weekday'] = pd.Categorical(weekday_fraud['weekday'], 
                                                   categories=weekday_order, 
                                                   ordered=True)
        weekday_fraud = weekday_fraud.sort_values('weekday')
        
        weekday_fraud.to_csv(output_dir / '04_fraud_by_weekday.csv', index=False)
    
    # --- 3. Fraud by Month (if available) ---
    monthly_fraud = None
    if 'trans_month' in df.columns:
        monthly_fraud = df.groupby('trans_month').agg({
            'is_fraud': ['sum', 'count', 'mean']
        }).reset_index()
        monthly_fraud.columns = ['month', 'fraud_count', 'total_count', 'fraud_rate']
        monthly_fraud['fraud_rate'] = (monthly_fraud['fraud_rate'] * 100).round(3)
        
        monthly_fraud.to_csv(output_dir / '04_fraud_by_month.csv', index=False)
    
    # --- 4. Text Report ---
    report_path = output_dir / '04_temporal_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - TEMPORAL ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. FRAUD BY HOUR OF DAY\n")
        f.write("-" * 70 + "\n")
        f.write(hourly_fraud.to_string(index=False) + "\n\n")
        
        # Find peak hours
        peak_hour = hourly_fraud.loc[hourly_fraud['fraud_rate'].idxmax()]
        low_hour = hourly_fraud.loc[hourly_fraud['fraud_rate'].idxmin()]
        
        f.write("KEY INSIGHTS:\n")
        f.write(f"Peak fraud hour:   {int(peak_hour['hour']):02d}:00 ({peak_hour['fraud_rate']:.3f}%)\n")
        f.write(f"Lowest fraud hour: {int(low_hour['hour']):02d}:00 ({low_hour['fraud_rate']:.3f}%)\n")
        
        # Night vs day analysis
        night_hours = hourly_fraud[hourly_fraud['hour'].isin([0,1,2,3,4,5])]['fraud_rate'].mean()
        day_hours = hourly_fraud[hourly_fraud['hour'].isin([9,10,11,12,13,14,15,16,17])]['fraud_rate'].mean()
        
        f.write(f"\nNight (00-05):     {night_hours:.3f}% fraud rate\n")
        f.write(f"Business (09-17):  {day_hours:.3f}% fraud rate\n")
        
        if night_hours > day_hours * 1.2:
            f.write("\n⚠️  NIGHT-TIME FRAUD PATTERN detected!\n")
            f.write("   → Fraudsters are more active at night\n")
            f.write("   → Consider time-based fraud rules\n")
        elif day_hours > night_hours * 1.2:
            f.write("\n⚠️  DAYTIME FRAUD PATTERN detected!\n")
            f.write("   → Fraud occurs during business hours\n")
        else:
            f.write("\n✓ Fraud distributed evenly across hours\n")
        
        f.write("\n")
        
        if weekday_fraud is not None:
            f.write("2. FRAUD BY DAY OF WEEK\n")
            f.write("-" * 70 + "\n")
            f.write(weekday_fraud.to_string(index=False) + "\n\n")
            
            peak_day = weekday_fraud.loc[weekday_fraud['fraud_rate'].idxmax()]
            f.write(f"Peak fraud day: {peak_day['weekday']} ({peak_day['fraud_rate']:.3f}%)\n\n")
        
        if monthly_fraud is not None:
            f.write("3. FRAUD BY MONTH\n")
            f.write("-" * 70 + "\n")
            f.write(monthly_fraud.to_string(index=False) + "\n\n")
    
    # --- 5. Visualizations ---
    n_plots = 2 + (1 if weekday_fraud is not None else 0) + (1 if monthly_fraud is not None else 0)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Plot 1: Fraud rate by hour (line plot)
    ax1 = axes[0]
    ax1.plot(hourly_fraud['hour'], hourly_fraud['fraud_rate'], 
             marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.fill_between(hourly_fraud['hour'], hourly_fraud['fraud_rate'], 
                     alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.3)
    
    # Highlight night hours
    ax1.axvspan(0, 6, alpha=0.1, color='navy', label='Night (00-06)')
    ax1.axvspan(22, 24, alpha=0.1, color='navy')
    ax1.legend()
    
    # Plot 2: Fraud count by hour (bar chart)
    ax2 = axes[1]
    bars = ax2.bar(hourly_fraud['hour'], hourly_fraud['fraud_count'], 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Frauds', fontsize=12, fontweight='bold')
    ax2.set_title('Fraud Count by Hour', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Weekday analysis (if available)
    if weekday_fraud is not None:
        ax3 = axes[2]
        colors = ['#3498db' if day in ['Saturday', 'Sunday'] else '#2ecc71' 
                 for day in weekday_fraud['weekday']]
        bars = ax3.bar(range(len(weekday_fraud)), weekday_fraud['fraud_rate'],
                      color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Fraud Rate by Day of Week', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(weekday_fraud)))
        ax3.set_xticklabels([day[:3] for day in weekday_fraud['weekday']], rotation=45)
        ax3.grid(axis='y', alpha=0.3)
    else:
        axes[2].axis('off')
    
    # Plot 4: Monthly analysis (if available) or heatmap
    if monthly_fraud is not None:
        ax4 = axes[3]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        bars = ax4.bar(monthly_fraud['month'], monthly_fraud['fraud_rate'],
                      color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Fraud Rate by Month', fontsize=14, fontweight='bold')
        ax4.set_xticks(monthly_fraud['month'])
        ax4.set_xticklabels([month_names[int(m)-1] for m in monthly_fraud['month']], rotation=45)
        ax4.grid(axis='y', alpha=0.3)
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Console Output ---
    print("⏰ Temporal Analysis Results:")
    print()
    print("📊 Fraud by Hour:")
    print(f"   Peak hour:   {int(peak_hour['hour']):02d}:00 ({peak_hour['fraud_rate']:.3f}%)")
    print(f"   Lowest hour: {int(low_hour['hour']):02d}:00 ({low_hour['fraud_rate']:.3f}%)")
    print()
    print(f"   Night (00-05):    {night_hours:.3f}%")
    print(f"   Business (09-17): {day_hours:.3f}%")
    print()
    
    if night_hours > day_hours * 1.2:
        print("⚠️  Night-time fraud pattern detected!")
    
    if weekday_fraud is not None:
        peak_day = weekday_fraud.loc[weekday_fraud['fraud_rate'].idxmax()]
        print(f"📅 Peak fraud day: {peak_day['weekday']} ({peak_day['fraud_rate']:.3f}%)")
        print()
    
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 04_fraud_by_hour.csv")
    if weekday_fraud is not None:
        print(f"   - 04_fraud_by_weekday.csv")
    if monthly_fraud is not None:
        print(f"   - 04_fraud_by_month.csv")
    print(f"   - 04_temporal_patterns.png")
    print(f"   - 04_temporal_analysis_report.txt")
    
    end = time.time()
    print(f"\n⏱️  Runtime: {end - start:.2f} seconds\n")
    
    return df


def run_chunked(input_path: Path, output_dir: Path, chunksize: int = 1_000_000):
    """
    Analyze temporal patterns on large CSV files using chunked processing.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save reports
        chunksize: Number of rows to process at a time
    """
    start = time.time()
    
    print(f"⏰ Starting chunked temporal analysis")
    print(f"   Input: {input_path}")
    print(f"   Chunk size: {chunksize:,} rows")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Accumulate counts ---
    print("  Scanning data...")
    
    # Accumulators
    hourly_counts = {h: {'fraud': 0, 'total': 0} for h in range(24)}
    weekday_counts = {}
    monthly_counts = {}
    
    chunk_num = 0
    total_rows = 0
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        chunk_num += 1
        total_rows += len(chunk)
        
        if 'is_fraud' not in chunk.columns or 'trans_hour' not in chunk.columns:
            continue
        
        # Accumulate hourly stats
        for hour in range(24):
            hour_data = chunk[chunk['trans_hour'] == hour]
            hourly_counts[hour]['fraud'] += (hour_data['is_fraud'] == 1).sum()
            hourly_counts[hour]['total'] += len(hour_data)
        
        # Accumulate weekday stats
        if 'trans_weekday' in chunk.columns:
            for weekday in chunk['trans_weekday'].unique():
                if pd.isna(weekday):
                    continue
                if weekday not in weekday_counts:
                    weekday_counts[weekday] = {'fraud': 0, 'total': 0}
                
                weekday_data = chunk[chunk['trans_weekday'] == weekday]
                weekday_counts[weekday]['fraud'] += (weekday_data['is_fraud'] == 1).sum()
                weekday_counts[weekday]['total'] += len(weekday_data)
        
        # Accumulate monthly stats
        if 'trans_month' in chunk.columns:
            for month in chunk['trans_month'].unique():
                if pd.isna(month):
                    continue
                if month not in monthly_counts:
                    monthly_counts[month] = {'fraud': 0, 'total': 0}
                
                month_data = chunk[chunk['trans_month'] == month]
                monthly_counts[month]['fraud'] += (month_data['is_fraud'] == 1).sum()
                monthly_counts[month]['total'] += len(month_data)
        
        if chunk_num % 5 == 0:
            print(f"    Scanned {total_rows:,} rows...")
    
    print(f"    Total rows scanned: {total_rows:,}\n")
    
    # --- Generate reports ---
    
    # Hourly report
    hourly_fraud = pd.DataFrame([
        {
            'hour': h,
            'fraud_count': hourly_counts[h]['fraud'],
            'total_count': hourly_counts[h]['total'],
            'fraud_rate': (hourly_counts[h]['fraud'] / hourly_counts[h]['total'] * 100) 
                         if hourly_counts[h]['total'] > 0 else 0
        }
        for h in range(24)
    ])
    hourly_fraud['fraud_rate'] = hourly_fraud['fraud_rate'].round(3)
    hourly_fraud.to_csv(output_dir / '04_fraud_by_hour.csv', index=False)
    
    # Weekday report
    weekday_fraud = None
    if weekday_counts:
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_fraud = pd.DataFrame([
            {
                'weekday': day,
                'fraud_count': weekday_counts[day]['fraud'],
                'total_count': weekday_counts[day]['total'],
                'fraud_rate': (weekday_counts[day]['fraud'] / weekday_counts[day]['total'] * 100)
                             if weekday_counts[day]['total'] > 0 else 0
            }
            for day in weekday_order if day in weekday_counts
        ])
        weekday_fraud['fraud_rate'] = weekday_fraud['fraud_rate'].round(3)
        weekday_fraud.to_csv(output_dir / '04_fraud_by_weekday.csv', index=False)
    
    # Monthly report
    monthly_fraud = None
    if monthly_counts:
        monthly_fraud = pd.DataFrame([
            {
                'month': month,
                'fraud_count': monthly_counts[month]['fraud'],
                'total_count': monthly_counts[month]['total'],
                'fraud_rate': (monthly_counts[month]['fraud'] / monthly_counts[month]['total'] * 100)
                             if monthly_counts[month]['total'] > 0 else 0
            }
            for month in sorted(monthly_counts.keys())
        ])
        monthly_fraud['fraud_rate'] = monthly_fraud['fraud_rate'].round(3)
        monthly_fraud.to_csv(output_dir / '04_fraud_by_month.csv', index=False)
    
    # Text report
    peak_hour = hourly_fraud.loc[hourly_fraud['fraud_rate'].idxmax()]
    low_hour = hourly_fraud.loc[hourly_fraud['fraud_rate'].idxmin()]
    
    night_hours = hourly_fraud[hourly_fraud['hour'].isin([0,1,2,3,4,5])]['fraud_rate'].mean()
    day_hours = hourly_fraud[hourly_fraud['hour'].isin([9,10,11,12,13,14,15,16,17])]['fraud_rate'].mean()
    
    report_path = output_dir / '04_temporal_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FRAUD DETECTION - TEMPORAL ANALYSIS (FULL DATASET)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. FRAUD BY HOUR OF DAY\n")
        f.write("-" * 70 + "\n")
        f.write(hourly_fraud.to_string(index=False) + "\n\n")
        
        f.write("KEY INSIGHTS:\n")
        f.write(f"Peak fraud hour:   {int(peak_hour['hour']):02d}:00 ({peak_hour['fraud_rate']:.3f}%)\n")
        f.write(f"Lowest fraud hour: {int(low_hour['hour']):02d}:00 ({low_hour['fraud_rate']:.3f}%)\n")
        f.write(f"\nNight (00-05):     {night_hours:.3f}% fraud rate\n")
        f.write(f"Business (09-17):  {day_hours:.3f}% fraud rate\n")
        
        if night_hours > day_hours * 1.2:
            f.write("\n⚠️  NIGHT-TIME FRAUD PATTERN detected!\n")
            f.write("   → Fraudsters are more active at night\n")
        
        if weekday_fraud is not None:
            f.write("\n2. FRAUD BY DAY OF WEEK\n")
            f.write("-" * 70 + "\n")
            f.write(weekday_fraud.to_string(index=False) + "\n\n")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Plot 1: Fraud rate by hour
    ax1 = axes[0]
    ax1.plot(hourly_fraud['hour'], hourly_fraud['fraud_rate'],
             marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax1.fill_between(hourly_fraud['hour'], hourly_fraud['fraud_rate'],
                     alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Fraud Rate by Hour', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.axvspan(0, 6, alpha=0.1, color='navy', label='Night')
    ax1.axvspan(22, 24, alpha=0.1, color='navy')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Fraud count by hour
    ax2 = axes[1]
    ax2.bar(hourly_fraud['hour'], hourly_fraud['fraud_count'],
           color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraud Count', fontsize=12, fontweight='bold')
    ax2.set_title('Fraud Count by Hour', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Weekday
    if weekday_fraud is not None:
        ax3 = axes[2]
        colors = ['#3498db' if day in ['Saturday', 'Sunday'] else '#2ecc71'
                 for day in weekday_fraud['weekday']]
        ax3.bar(range(len(weekday_fraud)), weekday_fraud['fraud_rate'],
               color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Fraud Rate by Weekday', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(weekday_fraud)))
        ax3.set_xticklabels([day[:3] for day in weekday_fraud['weekday']], rotation=45)
        ax3.grid(axis='y', alpha=0.3)
    else:
        axes[2].axis('off')
    
    # Plot 4: Monthly
    if monthly_fraud is not None:
        ax4 = axes[3]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax4.bar(monthly_fraud['month'], monthly_fraud['fraud_rate'],
               color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Fraud Rate by Month', fontsize=14, fontweight='bold')
        ax4.set_xticks(monthly_fraud['month'])
        ax4.set_xticklabels([month_names[int(m)-1] for m in monthly_fraud['month']], rotation=45)
        ax4.grid(axis='y', alpha=0.3)
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Console output - DETAILED HOURLY BREAKDOWN
    print("⏰ Temporal Analysis Results:")
    print()
    print("=" * 80)
    print("FRAUD RATE BY HOUR (All 24 Hours)")
    print("=" * 80)
    print(f"{'Hour':<8} {'Fraud Count':>15} {'Total Trans':>15} {'Fraud Rate':>15}")
    print("-" * 80)
    
    for _, row in hourly_fraud.iterrows():
        hour = int(row['hour'])
        fraud_count = int(row['fraud_count'])
        total_count = int(row['total_count'])
        fraud_rate = row['fraud_rate']
        
        # Time period label
        if 0 <= hour < 6:
            period = "🌙 Night"
        elif 6 <= hour < 12:
            period = "🌅 Morning"
        elif 12 <= hour < 18:
            period = "☀️  Afternoon"
        else:
            period = "🌆 Evening"
        
        print(f"{hour:02d}:00 {period:<12} {fraud_count:>12,} {total_count:>15,} {fraud_rate:>13.3f}%")
    
    print("=" * 80)
    print()
    
    # Summary statistics
    print("📊 SUMMARY BY TIME PERIOD:")
    print("-" * 80)
    
    night = hourly_fraud[hourly_fraud['hour'].isin([0,1,2,3,4,5])]
    morning = hourly_fraud[hourly_fraud['hour'].isin([6,7,8,9,10,11])]
    afternoon = hourly_fraud[hourly_fraud['hour'].isin([12,13,14,15,16,17])]
    evening = hourly_fraud[hourly_fraud['hour'].isin([18,19,20,21,22,23])]
    
    print(f"🌙 Night (00-05):      {night['fraud_rate'].mean():.3f}% avg fraud rate")
    print(f"🌅 Morning (06-11):    {morning['fraud_rate'].mean():.3f}% avg fraud rate")
    print(f"☀️  Afternoon (12-17):  {afternoon['fraud_rate'].mean():.3f}% avg fraud rate")
    print(f"🌆 Evening (18-23):    {evening['fraud_rate'].mean():.3f}% avg fraud rate")
    print()
    
    print(f"⚠️  Peak hour:   {int(peak_hour['hour']):02d}:00 ({peak_hour['fraud_rate']:.3f}%)")
    print(f"✅ Lowest hour: {int(low_hour['hour']):02d}:00 ({low_hour['fraud_rate']:.3f}%)")
    print()
    
    # Pattern detection
    if night['fraud_rate'].mean() > morning['fraud_rate'].mean() * 1.2:
        print("⚠️  NIGHT-TIME FRAUD PATTERN detected!")
        print("   → Fraudsters are significantly more active at night")
    elif afternoon['fraud_rate'].mean() > night['fraud_rate'].mean() * 1.2:
        print("⚠️  DAYTIME FRAUD PATTERN detected!")
        print("   → Fraud occurs more during afternoon hours")
    else:
        print("✓ Fraud distributed relatively evenly across time periods")
    
    print()
    
    if weekday_fraud is not None:
        print("📅 WEEKDAY ANALYSIS:")
        print("-" * 80)
        for _, row in weekday_fraud.iterrows():
            day = row['weekday']
            fraud_rate = row['fraud_rate']
            fraud_count = int(row['fraud_count'])
            emoji = "🎉" if day in ['Saturday', 'Sunday'] else "💼"
            print(f"{emoji} {day:<10} {fraud_count:>12,} frauds  ({fraud_rate:.3f}%)")
        print()
    
    end = time.time()
    
    print("=" * 80)
    print(f"💾 Reports saved to: {output_dir}")
    print(f"   - 04_fraud_by_hour.csv")
    if weekday_fraud is not None:
        print(f"   - 04_fraud_by_weekday.csv")
    if monthly_fraud is not None:
        print(f"   - 04_fraud_by_month.csv")
    print(f"   - 04_temporal_patterns.png")
    print(f"   - 04_temporal_analysis_report.txt")
    print(f"\n⏱️  Runtime: {end - start:.1f} seconds ({(end - start)/60:.1f} minutes)\n")


# Standalone execution
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports" / "eda_reports"
    
    input_file = data_dir / "03_transformed.csv"
    
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        print("   Please run the data preparation pipeline first.")
        exit(1)
    
    print(f"Processing: {input_file}")
    
    # Use chunked processing
    run_chunked(input_file, reports_dir, chunksize=1_000_000)
    
    print("✅ Temporal analysis complete (full dataset)!")

