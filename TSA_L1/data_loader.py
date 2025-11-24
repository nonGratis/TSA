import pandas as pd
import io
import requests
from pathlib import Path
from datetime import datetime

def fetch_data(url):
    if "pubhtml" in url:
        base_url = url.split("pubhtml")[0] + "pub"
        if "?" in url:
            params = url.split("?")[1]
            csv_url = f"{base_url}?{params}&output=csv"
        else:
            csv_url = f"{base_url}?output=csv"
    else:
        csv_url = url
    
    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        _save_raw_data(df)
        
        return df

    except Exception as e:
        print(f"Помилка парсингу: {e}")
        return None

def _save_raw_data(df):
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{timestamp}.csv'
    filepath = data_dir / filename
    
    df.to_csv(filepath, index=False)
    print(f"Дані збережено: {filepath}")