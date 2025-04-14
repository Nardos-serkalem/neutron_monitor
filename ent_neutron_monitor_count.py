import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from io import StringIO
import re

# ---- CONFIGURATION ----
data_base_url = "http://196.188.116.60/Min_Data/"  # ← CHANGE this
username = "user"                  # ← your credentials
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

# ---- STEP 3: LOAD & CONCATENATE ----
auth = HTTPBasicAuth(username, password)
file_urls = get_csv_filenames_from_url(data_base_url, auth)
dfs = [read_and_process_csv_from_url(url, auth) for url in file_urls]
df_all = pd.concat(dfs).sort_index()

# 🔐 Remove duplicates before resampling
df_all = df_all[~df_all.index.duplicated(keep='first')]
# Resample and interpolate
#df_all = df_all.resample("1T").interpolate("linear")

# ---- STEP 4: PLOT SCATTER ----
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

'''
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np

# ---- FILE NAME ----
filename = "2025049_min_Ethiopia.csv"

# ---- EXTRACT YEAR AND DOY FROM FILENAME ----
year = int(filename.split('_')[0][:4])       # 2025
doy = int(filename.split('_')[0][4:7])       # 092

# ---- READ CSV FILE ----
# Adjust the delimiter if needed (e.g., tabs or spaces)
df = pd.read_csv(filename, sep=r'\s+|,', engine='python', header=None,
                 names=["Time", "Neutron_Count", "Altitude", "Temperature", "Pressure"])
# ---- REPLACE ZERO VALUES WITH NaN (exclude Time column) ----
df = df.replace(0, np.nan)

# ---- CONVERT TIME TO DATETIME ----
# Create base date from year and day of year
base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)

# Convert HH:MM to full datetime
df['DateTime'] = df['Time'].apply(lambda t: base_date + timedelta(
    hours=int(t.split(":")[0]), minutes=int(t.split(":")[1])))

# ---- SET INDEX TO DATETIME ----
df.set_index('DateTime', inplace=True)

# ---- PLOT TIME SERIES ----
plt.figure(figsize=(15, 10))

# Neutron Count
plt.subplot(4, 1, 1)
plt.scatter(df.index, df['Neutron_Count'], s=5, color='blue')
plt.title('Neutron Count')
plt.ylabel('Counts')

# Altitude
plt.subplot(4, 1, 2)
plt.scatter(df.index, df['Altitude'], s=5, color='green')
plt.title('Altitude')
plt.ylabel('m')

# Temperature
plt.subplot(4, 1, 3)
plt.scatter(df.index, df['Temperature'], s=5, color='orange')
plt.title('Temperature')
plt.ylabel('°C')

# Pressure
plt.subplot(4, 1, 4)
plt.scatter(df.index, df['Pressure'], s=5, color='purple')
plt.title('Pressure')
plt.ylabel('hPa')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
'''
