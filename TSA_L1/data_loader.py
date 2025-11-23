import pandas as pd
import io
import requests

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

        return df

    except Exception as e:
        print(f"Помилка парсингу: {e}")
        return None