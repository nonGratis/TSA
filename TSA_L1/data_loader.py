import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import re

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        
        for script in soup.find_all('script'):
            if script.string and 'pageUrl:' in script.string:
                match = re.search(r'pageUrl:\s*"([^"]+)"', script.string) # Витяг посилання на таблицю
                if match:
                    sheet_url = match.group(1).replace(r'\/', '/')
                    return _parse_sheet(sheet_url)
        
        print("Помилка: не знайдено посилання на sheet")
        return None
        
    except Exception as e:
        print(f"Помилка: {e}")
        return None

def _parse_sheet(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='waffle')
        
        if not table:
            print("Помилка: таблицю не знайдено")
            
            return None
        
        all_rows = table.find_all('tr')
        data = []
        
        for row in all_rows:
            cols = [td.get_text(strip=True) for td in row.find_all('td')]
            if any(cols):
                data.append(cols)        
        df = pd.DataFrame(data[1:], columns=data[0]) # перрший як заголовки
        
        if 'r_id' in df.columns:
            df['r_id'] = pd.to_numeric(df['r_id'], errors='coerce')        
        _save_data(df)
        return df
        
    except Exception as e:
        print(f"Помилка парсингу: {e}")
        return None

def _save_data(df):
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = data_dir / f'{timestamp}.csv'
    
    df.to_csv(filepath, index=False)
    print(f"Дані збережено: {filepath}")