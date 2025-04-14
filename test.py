import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from io import StringIO
import re
from scipy.stats import zscore

# ---- CONFIGURATION ----
data_base_url = "http://196.188.116.60/Min_Data/"
username = "user"
password = "@Ssgi*123"
days_to_fetch = 3

# ---- STEP 1: GET FILE LIST FROM AUTHENTICATED URL ----
def get_csv_filenames_from_url(base_url, auth):
    response = requests.get(base_url, auth=auth)
    file_list = re.findall(r'(\d{7}_min_Ethiopia\.csv)', response.text)
    file_list = sorted(file_list)[-days_to_fetch:]
    return [base_url + fname for fname in file_list]

# ---- STEP 2: READ & PROCESS EACH FILE ----
def read_and_process_csv_from_url(file_url, auth):
    response = requests.get(file_url, auth=auth)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text), sep=r'\s+|,', engine='python', header=None,
                         names=["Time", "Neutron_Count", "Altitude", "Temperature", "Pressure"])
        
        # Extract year and DOY from filename
        match = re.search(r'(\d{4})(\d{3})_min_Ethiopia\.csv', file_url)
        if match:
            year = int(match.group(1))
            doy = int(match.group(2))
            base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        else:
            raise ValueError("Filename format does not match expected pattern.")

        df['DateTime'] = df['Time'].apply(lambda t: base_date + timedelta(
            hours=int(t.split(":")[0]), minutes=int(t.split(":")[1])))

        df.set_index('DateTime', inplace=True)
        df.drop(columns=['Time'], inplace=True)
        df = df.replace(0, np.nan)  # Replace 0s with NaN
        return df
    else:
        print(f"Failed to download: {file_url} (Status: {response.status_code})")
        return pd.DataFrame()

# ---- Z-SCORE OUTLIER FILTER ----
def remove_outliers_zscore(df, threshold=3):
    df_filtered = df.copy()
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(zscore(df_filtered[numeric_cols], nan_policy='omit'))
    mask = (z_scores < threshold).all(axis=1)
    return df_filtered[mask]

# ---- STEP 3: LOAD & CONCATENATE ----
auth = HTTPBasicAuth(username, password)
file_urls = get_csv_filenames_from_url(data_base_url, auth)
dfs = [read_and_process_csv_from_url(url, auth) for url in file_urls]
df_all = pd.concat(dfs).sort_index()
df_all = df_all[~df_all.index.duplicated(keep='first')]

# ---- STEP 4: OUTLIER REMOVAL ----
df_all = remove_outliers_zscore(df_all)

# ---- STEP 5: PLOT TIME SERIES ----
plt.figure(figsize=(15, 10))

plt.subplot(4, 1, 1)
plt.scatter(df_all.index, df_all['Neutron_Count'], s=5, color='blue')
plt.title('Neutron Count')
plt.ylabel('Counts')

plt.subplot(4, 1, 2)
plt.scatter(df_all.index, df_all['Altitude'], s=5, color='green')
plt.title('Altitude')
plt.ylabel('m')

plt.subplot(4, 1, 3)
plt.scatter(df_all.index, df_all['Temperature'], s=5, color='orange')
plt.title('Temperature')
plt.ylabel('°C')

plt.subplot(4, 1, 4)
plt.scatter(df_all.index, df_all['Pressure'], s=5, color='purple')
plt.title('Pressure')
plt.ylabel('hPa')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
