import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import re
import os
import tempfile
from datetime import datetime, timedelta, timezone
from matplotlib.dates import DateFormatter, HourLocator
from requests.auth import HTTPBasicAuth

DATA_BASE_URL = "http://196.188.116.60/Min_Data/"
USERNAME = "user"
PASSWORD = "@Ssgi*123"
DAYS_TO_FETCH = 3
Z_SCORE_THRESHOLD = 3
WINDOW_SIZE = 60 # this one is for outlier detection in each 60min 
UPDATE_INTERVAL = '5min' 

class NeutronDataProcessor:
    def __init__(self):
        self.auth = HTTPBasicAuth(USERNAME, PASSWORD)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.target_files = self._get_target_files()
        self.df = self._process_data()

    def _get_target_dates(self):
        today = datetime.now(timezone.utc)
        return [(today - timedelta(days=i)).strftime("%Y%j") for i in range(DAYS_TO_FETCH)]

    def _get_target_files(self):
        target_dates = self._get_target_dates()
        response = requests.get(DATA_BASE_URL, auth=self.auth)
        all_files = re.findall(r'(\d{7}_min_Ethiopia\.csv)', response.text)
        return [f for f in all_files if f[:7] in target_dates]

    def _download_file(self, filename):
        url = f"{DATA_BASE_URL}{filename}"
        response = requests.get(url, auth=self.auth)
        if response.status_code == 200:
            file_path = os.path.join(self.temp_dir.name, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        return None

    def _process_file(self, file_path):
        df = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None,
                         names=["Time", "Neutron_Count", "Temperature", "Pressure"])
        match = re.search(r'(\d{4})(\d{3})', os.path.basename(file_path))
        year, doy = int(match[1]), int(match[2])
        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        df['DateTime'] = df['Time'].apply(
            lambda t: base_date + timedelta(
                hours=int(str(int(t)).zfill(4)[:2]),
                minutes=int(str(int(t)).zfill(4)[2:])
            ) if pd.notna(t) else base_date
        )

        df = df.set_index('DateTime').drop(columns=['Time'])
        return df.replace(0, np.nan)

    def _remove_outliers(self, series):
        roll_mean = series.rolling(window=WINDOW_SIZE, min_periods=1).mean()
        roll_std = series.rolling(window=WINDOW_SIZE, min_periods=1).std().replace(0, 1)
        z_score = np.abs((series - roll_mean) / roll_std)
        return series.mask(z_score > Z_SCORE_THRESHOLD, roll_mean)

    def _process_data(self):
        dfs = []
        for filename in self.target_files:
            local_path = self._download_file(filename)
            if local_path:
                df = self._process_file(local_path)
                for col in df.columns:
                    df[col] = self._remove_outliers(df[col])
                dfs.append(df)

        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        #return combined.resample('1T').asfreq().interpolate()
        return combined.resample('1min').asfreq().interpolate()


    def plot_data(self):
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        params = [
            ('Neutron_Count', 'blue', 'Neutron Count (CPM)'),
            ('Temperature', 'orange', 'Temperature (°C)'),
            ('Pressure', 'purple', 'Pressure (hPa)')
        ]

        for idx, (col, color, title) in enumerate(params):
            axs[idx].scatter(self.df.index, self.df[col], s=5, color=color, alpha=0.7)
            axs[idx].set_ylabel(title)
            axs[idx].xaxis.set_major_locator(HourLocator(interval=6))
            axs[idx].xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
            axs[idx].grid(True, linestyle=':', alpha=0.5)

        plt.xlabel('Time (UTC)')
        plt.tight_layout()
        plt.show()

    def __del__(self):
        self.temp_dir.cleanup()

if __name__ == "__main__":
    processor = NeutronDataProcessor()
    processor.plot_data()
