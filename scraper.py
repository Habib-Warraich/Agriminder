import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def get_latest_rates():
    url = "https://agripunjab.gov.pk/pricelist"
    try:
        # Fetching data from the Punjab Govt site
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Finding the price table
        table = soup.find('table') 
        data = []
        
        # Extracting rows
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                crop = cols[0].text.strip()
                price = cols[1].text.strip()
                data.append({"Crop": crop, "Price": price})
        
        df = pd.DataFrame(data)
        # Save to local CSV so the app works offline too
        df.to_csv('latest_rates.csv', index=False)
        return df
    except:
        # If no internet, load the last saved data
        if os.path.exists('latest_rates.csv'):
            return pd.read_csv('latest_rates.csv')
        return None