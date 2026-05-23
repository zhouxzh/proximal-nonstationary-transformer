import numpy as np
import os 
import re
from tqdm import tqdm
import pandas as pd

from sklearn.preprocessing import StandardScaler



def calculate_metrics(y_true, y_pred):
    """
    合并计算所有指标：RMSE、MAE、sMAPE、R2
    返回一个字典
    """
    # RMSE
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # MAE
    mae_val = np.mean(np.abs(y_true - y_pred))
    # sMAPE
    smape_val = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_val = 1 - (ss_res / ss_tot)
    
    return {
        'rmse': rmse_val,
        'mae': mae_val,
        'smape': smape_val,
        'r2': r2_val
    }
    
standard_scaler = StandardScaler()
data = pd.read_csv('dataset/GLD.csv')
data.set_index('date', inplace=True)
data_values = data['Close'].values.reshape(-1, 1)
standard_scaler.fit(data_values)

metrics_list = []

for f in tqdm(os.listdir('results')):
    if os.path.isdir(os.path.join('results', f)):
        pattern = re.compile(r'''
            ^
            long_term_forecast_
            (?P<model_id>.+?)_
            GLD_
            (?P<model>.+?)_
            custom_
            ft(?P<features>[^_]+)_
            sl(?P<seq_len>\d+)_
            ll(?P<label_len>\d+)_
            pl(?P<pred_len>\d+)_
            dm(?P<d_model>\d+)_
            nh(?P<n_heads>\d+)_
            el(?P<e_layers>\d+)_
            dl(?P<d_layers>\d+)_
            df(?P<d_ff>\d+)_
            fc(?P<factor>\d+)_
            eb(?P<embed>[^_]+)_
            dt(?P<distil>[^_]+)_
            (?P<des>[^_]+)_
            (?P<ii>[^_]+)
            $
        ''', re.VERBOSE)

        m = pattern.match(f)
        if m:
            params = m.groupdict()
            # print(params)
        else:
            print(f"Folder name '{f}' doesn't match expected pattern.")
            continue
        pred = np.load(os.path.join('results', f, 'pred.npy'))
        true = np.load(os.path.join('results', f, 'true.npy'))
        pred = pred[:, -1, 0]
        true = true[:, -1, 0]
        pred = standard_scaler.inverse_transform(pred.reshape(-1, 1))
        true = standard_scaler.inverse_transform(true.reshape(-1, 1))
        metrics = calculate_metrics(true, pred)
        metrics_list.append({**params, **metrics})

metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.sort_values(by=['model', 'seq_len', 'pred_len'])
metrics_df.to_excel('metrics_summary.xlsx', index=False)
