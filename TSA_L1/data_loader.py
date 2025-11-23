import pandas as pd

def fetch_data(url: str) -> pd.DataFrame:       
    df = pd.read_csv(url)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')

    return df[['timestamp', 'r_id']]

