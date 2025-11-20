#!/usr/bin/env python3
"""
Visualize Optuna Optimization Results
======================================

This script loads the Optuna optimization history and creates
visualizations to understand the hyperparameter optimization process.

Visualizations created:
1. Optimization History - ROC-AUC score over trials
2. Parameter Importance - Which hyperparameters matter most
3. Convergence Plot - Best score over time

Input:
- models/xgboost_optimized/optimization_history.csv

Output:
- reports/evaluation/figures/optimization_history.png
- reports/evaluation/figures/parameter_importance.png
- reports/evaluation/figures/convergence_plot.png

"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_optimization_history(history_df: pd.DataFrame, output_path: Path):
    """
    Plot ROC-AUC score over trials.
    
    Shows how the optimization improved over time.
    """
    print("📊 Creating optimization history plot...")
    
    trial_numbers = history_df['number'].values
    values = history_df['value'].values
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all trials
    ax.scatter(trial_numbers, values, alpha=0.6, s=50, c='steelblue', label='Trial')
    
    # Plot best score line
    best_scores = []
    current_best = -np.inf
    for v in values:
        current_best = max(current_best, v)
        best_scores.append(current_best)
    
    ax.plot(trial_numbers, best_scores, 'r-', linewidth=2, label='Best Score', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Optuna Optimization History\nBayesian Hyperparameter Search', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add best score annotation
    best_idx = np.argmax(values)
    best_trial = trial_numbers[best_idx]
    best_value = values[best_idx]
    
    ax.annotate(f'Best: {best_value:.4f}\n(Trial {best_trial})',
                xy=(best_trial, best_value),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_parameter_importance(history_df: pd.DataFrame, output_path: Path):
    """
    Plot parameter importance.
    
    Shows which hyperparameters had the most impact on performance.
    """
    print("📊 Creating parameter importance plot...")
    
    try:
        # Get parameter columns (those starting with 'params_')
        param_cols = [col for col in history_df.columns if col.startswith('params_')]
        
        if not param_cols:
            print("⚠️  No parameter columns found")
            return
        
        # Calculate correlation between each parameter and ROC-AUC
        importances = {}
        
        for param_col in param_cols:
            param_name = param_col.replace('params_', '')
            
            # Get non-null values
            mask = history_df[param_col].notna() & history_df['value'].notna()
            
            if mask.sum() > 1:
                correlation = np.corrcoef(
                    history_df.loc[mask, param_col], 
                    history_df.loc[mask, 'value']
                )[0, 1]
                
                if not np.isnan(correlation):
                    importances[param_name] = abs(correlation)
        
        if not importances:
            print("⚠️  Not enough data to calculate parameter importance")
            return
        
        # Sort by importance
        sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        params, importance_values = zip(*sorted_params)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))
        bars = ax.barh(range(len(params)), importance_values, color=colors)
        
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params, fontsize=10)
        ax.set_xlabel('Importance (Absolute Correlation)', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Importance\nWhich parameters matter most?', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_values)):
            ax.text(val + 0.01, i, f'{val:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create parameter importance plot: {e}")


def plot_convergence(history_df: pd.DataFrame, output_path: Path):
    """
    Plot convergence - best score over time.
    
    Shows how quickly the optimization converged to the best solution.
    """
    print("📊 Creating convergence plot...")
    
    values = history_df['value'].values
    
    if len(values) == 0:
        print("⚠️  No valid trials to plot")
        return
    
    # Calculate cumulative best
    best_scores = []
    current_best = -np.inf
    for v in values:
        current_best = max(current_best, v)
        best_scores.append(current_best)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot convergence
    ax.plot(range(1, len(best_scores) + 1), best_scores, 
           'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    
    # Fill area under curve
    ax.fill_between(range(1, len(best_scores) + 1), best_scores, 
                    alpha=0.3, color='steelblue')
    
    # Formatting
    ax.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best ROC-AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Convergence\nBest score improvement over trials', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add final score annotation
    final_score = best_scores[-1]
    ax.axhline(y=final_score, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(len(best_scores) * 0.5, final_score + 0.0001, 
           f'Final Best: {final_score:.4f}',
           ha='center', va='bottom', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION VISUALIZATION")
    print("=" * 70)
    print()
    
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    history_path = project_root / 'models' / 'xgboost_optimized' / 'optimization_history.csv'
    output_dir = project_root / 'reports' / 'evaluation' / 'figures'
    
    # Check if history exists
    if not history_path.exists():
        print(f"❌ ERROR: Optimization history not found!")
        print(f"   Expected: {history_path}")
        print()
        print("💡 Make sure you've run:")
        print("   python src/modeling/04_train_selected_features.py")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load history
    print(f"📂 Loading optimization history...")
    history_df = pd.read_csv(history_path)
    print(f"✓ Loaded {len(history_df)} trials")
    print(f"✓ Best ROC-AUC: {history_df['value'].max():.4f}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    print()
    
    plot_optimization_history(history_df, output_dir / 'optimization_history.png')
    plot_parameter_importance(history_df, output_dir / 'parameter_importance.png')
    plot_convergence(history_df, output_dir / 'convergence_plot.png')
    
    print()
    print("=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print()
    print("📁 Output files:")
    print(f"   {output_dir / 'optimization_history.png'}")
    print(f"   {output_dir / 'parameter_importance.png'}")
    print(f"   {output_dir / 'convergence_plot.png'}")
    print()


if __name__ == "__main__":
    main()
