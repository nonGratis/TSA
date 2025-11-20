import pandas as pd

def fetch_data(url: str) -> pd.DataFrame:       
    df = pd.read_csv(url)

    # Перейменовуємо 'r_id' на 'response_count' для сумісності з логікою аналізу.
    df.rename(columns={'r_id': 'response_count'}, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')

    return df[['timestamp', 'response_count']]

