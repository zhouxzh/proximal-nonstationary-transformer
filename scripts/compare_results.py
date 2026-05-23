import os
import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def parse_log_files(log_dir='./logs'):
    """
    Parses log files in the specified directory and extracts parameters and metrics.
    """
    data = []
    
    # Regex pattern for the filename
    # Pattern: ${model_name}_Glod_ft${feature}_sl${seq_len}_ll${label_len}_pl${pred_len}.log
    filename_pattern = re.compile(r'^(?P<model_name>.+)_GLD_ft(?P<feature>.+)_sl(?P<seq_len>\d+)_ll(?P<label_len>\d+)_pl(?P<pred_len>\d+)\.log$')
    
    # Regex pattern for the last line metrics
    # Pattern: MSE:0.0887+-nan, MAE:0.2199+-nan, ...
    metrics_pattern = re.compile(r'MSE:(?P<mse>[\d\.]+).*, MAE:(?P<mae>[\d\.]+).*')

    if not os.path.exists(log_dir):
        print(f"Directory {log_dir} not found.")
        return pd.DataFrame()

    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'):
            continue
            
        match_name = filename_pattern.match(filename)
        if match_name:
            params = match_name.groupdict()
            
            # Convert numeric strings to integers
            params['seq_len'] = int(params['seq_len'])
            params['label_len'] = int(params['label_len'])
            params['pred_len'] = int(params['pred_len'])
            
            filepath = os.path.join(log_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                        
                    # Check the last line for metrics
                    last_line = lines[-1].strip()
                    match_metrics = metrics_pattern.search(last_line)
                    
                    if match_metrics:
                        metrics = match_metrics.groupdict()
                        params['mse'] = float(metrics['mse'])
                        params['mae'] = float(metrics['mae'])
                        # Calculate RMSE from MSE
                        params['rmse'] = np.sqrt(params['mse'])
                        data.append(params)
                    else:
                        print(f"Warning: Could not parse metrics in {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return pd.DataFrame(data)

def plot_results(df):
    """
    Plots Pred_Len vs MAE and Pred_Len vs RMSE for different models.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Sort data for cleaner line plots
    df = df.sort_values(by=['model_name', 'pred_len'])
    
    models = df['model_name'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    
    # Plot 1: Pred_Len vs MAE
    for model in models:
        subset = df[df['model_name'] == model]
        axes[0].plot(subset['pred_len'], subset['mae'], label=model)
    
    axes[0].set_title('Prediction Length vs MAE')
    axes[0].set_xlabel('Prediction Length')
    axes[0].set_ylabel('MAE')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Pred_Len vs RMSE
    for model in models:
        subset = df[df['model_name'] == model]
        axes[1].plot(subset['pred_len'], subset['rmse'], label=model)
    
    axes[1].set_title('Prediction Length vs RMSE')
    axes[1].set_xlabel('Prediction Length')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    axes[1].grid(True)

    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    print("Plots saved to comparison_results.png")

if __name__ == "__main__":
    df_results = parse_log_files()
    
    if not df_results.empty:
        print("Extracted Data:")
        print(df_results[['model_name', 'pred_len', 'mse', 'mae', 'rmse']].head())
        plot_results(df_results)
    else:
        print("No valid log data found.")