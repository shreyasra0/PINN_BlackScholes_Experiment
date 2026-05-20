# preprocess.py
import pandas as pd
import numpy as np

def run_preprocessing(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    
    df.columns = df.columns.str.strip().str.replace('[', '', regex=False).str.replace(']', '', regex=False)
    
    for col in ['C_BID', 'C_ASK']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
            
    if 'STRIKE' in df.columns:
        df['STRIKE'] = pd.to_numeric(df['STRIKE'].astype(str).str.strip(), errors='coerce')

    df = df.dropna(subset=['UNDERLYING_LAST', 'DTE', 'QUOTE_DATE'])
    if 'C_BID' in df.columns and 'C_ASK' in df.columns:
        df = df.dropna(subset=['C_BID', 'C_ASK', 'STRIKE' if 'STRIKE' in df.columns else 'UNDERLYING_LAST'])
    
    daily_underlying = df.drop_duplicates(subset=['QUOTE_DATE']).sort_values('QUOTE_DATE')
    spot_prices = daily_underlying['UNDERLYING_LAST'].values.astype(np.float32)
    
    log_returns = np.log(spot_prices[1:] / spot_prices[:-1])
    sigma_historical = np.std(log_returns, ddof=1) * np.sqrt(252)
    
    df = df[df['DTE'] > 0]
    
    S = df['UNDERLYING_LAST'].values.astype(np.float32)
    t = (df['DTE'].values.astype(np.float32)) / 365.0
    K = df['STRIKE'].values.astype(np.float32) if 'STRIKE' in df.columns else df['UNDERLYING_LAST'].values * 0.95
    
    X_processed = np.stack([S, t, K], axis=1)
    
    bid = df['C_BID'].values.astype(np.float32) if 'C_BID' in df.columns else df['UNDERLYING_LAST'].values * 0.05
    ask = df['C_ASK'].values.astype(np.float32) if 'C_ASK' in df.columns else bid * 1.02
    y_processed = ((bid + ask) / 2.0).reshape(-1, 1)
    
    np.save("data/bs_X_processed.npy", X_processed)
    np.save("data/bs_y_processed.npy", y_processed)
    np.save("data/bs_meta_sigma.npy", np.array([sigma_historical], dtype=np.float32))

if __name__ == "__main__":
    run_preprocessing("data/qqq_2020_2022.csv")