import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, HourLocator
from datetime import datetime, timedelta
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from io import StringIO
import re
from collections import deque

# ---- CONFIGURATION ----
DATA_BASE_URL = "http://196.188.116.60/Min_Data/"
USERNAME = "user"
PASSWORD = "@Ssgi*123"
DAYS_TO_FETCH = 3
BUFFER_SIZE = 1440  # 24 hours of minute data
UPDATE_INTERVAL = 60000  # 60 seconds (1 minute)
Z_THRESHOLD = 2.5

class NeutronMonitorDashboard:
    def __init__(self):
        self.auth = HTTPBasicAuth(USERNAME, PASSWORD)
        self.colors = ['blue', 'orange', 'purple']
        self.live_data = deque(maxlen=BUFFER_SIZE)
        
        # Initialize plot
        self.fig, self.axs = plt.subplots(3, 1, figsize=(14, 8))
        self._configure_plots()
        self._initialize_data()

    def _configure_plots(self):
        self.lines = []
        time_format = DateFormatter('%m-%d %H:%M')
        
        parameters = [
            ('Neutron Count', 'Counts'),
            ('Temperature', '°C'),
            ('Pressure', 'hPa')
        ]
        
        for idx, (ax, (title, ylabel)) in enumerate(zip(self.axs, parameters)):
            ax.xaxis.set_major_formatter(time_format)
            ax.xaxis.set_major_locator(HourLocator(interval=4))
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylabel(ylabel, rotation=0, labelpad=20)
            ax.set_title(title, fontsize=10)
            
            # Initialize plot lines
            line, = ax.plot([], [], lw=1.5, color=self.colors[idx])
            self.lines.append(line)

    def _initialize_data(self):
        """Load initial historical data"""
        file_urls = self._fetch_file_list()
        dfs = [self._process_file(url) for url in file_urls]
        df = pd.concat(dfs).sort_index()
        df = df[~df.index.duplicated(keep='first')].replace(0, np.nan)
        
        # Populate initial buffer
        for ts, row in df.iterrows():
            self.live_data.append((ts, row['Neutron_Count'], row['Temperature'], row['Pressure']))

    def _fetch_file_list(self):
        response = requests.get(DATA_BASE_URL, auth=self.auth)
        file_list = re.findall(r'(\d{7}_min_Ethiopia\.csv)', response.text)
        return [DATA_BASE_URL + f for f in sorted(file_list)[-DAYS_TO_FETCH:]]

    def _process_file(self, file_url):
        response = requests.get(file_url, auth=self.auth)
        if response.status_code != 200:
            return pd.DataFrame()
        
        df = pd.read_csv(StringIO(response.text), sep=r'\s+|,', 
                        header=None, names=["Time", "Neutron_Count", 
                        "Altitude", "Temperature", "Pressure"])
        
        # Date parsing
        match = re.search(r'(\d{4})(\d{3})', file_url)
        base_date = datetime(int(match[1]), 1, 1) + timedelta(days=int(match[2])-1)
        df['DateTime'] = df['Time'].apply(lambda t: base_date + 
            timedelta(hours=int(t[:2]), minutes=int(t[3:])))
        
        return df.set_index('DateTime').drop(columns=['Time', 'Altitude'])

    def _update_data(self):
        """Check for and process new data files"""
        new_files = self._fetch_file_list()[-1:]  # Get latest file only
        for url in new_files:
            df = self._process_file(url)
            for ts, row in df.iterrows():
                if ts > self.live_data[-1][0] if self.live_data else True:
                    self.live_data.append((ts, row['Neutron_Count'], row['Temperature'], row['Pressure']))

    def _filter_outliers(self, values):
        """Robust outlier detection using IQR"""
        q1 = np.nanquantile(values, 0.25)
        q3 = np.nanquantile(values, 0.75)
        iqr = q3 - q1
        return np.where((values < (q1 - 1.5 * iqr)) | (values > (q3 + 1.5 * iqr)), np.nan, values)

    def _update_plot(self, frame):
        self._update_data()  # Refresh data
        
        # Extract data from buffer
        times = [item[0] for item in self.live_data]
        neutron = [item[1] for item in self.live_data]
        temp = [item[2] for item in self.live_data]
        pressure = [item[3] for item in self.live_data]
        
        artists = []
        for idx, (data, ax) in enumerate(zip([neutron, temp, pressure], self.axs)):
            # Filter outliers
            clean_data = self._filter_outliers(np.array(data))
            
            # Update plot
            self.lines[idx].set_data(times, clean_data)
            
            # Dynamic axis scaling
            valid_vals = clean_data[~np.isnan(clean_data)]
            if valid_vals.size > 0:
                ax.set_ylim(np.nanmin(valid_vals)*0.98, np.nanmax(valid_vals)*1.02)
                ax.set_xlim(max(times)-timedelta(hours=24), max(times))
            
            artists.append(self.lines[idx])
        
        return artists

    def start(self):
        ani = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=UPDATE_INTERVAL,
            blit=True,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()

if __name__ == "__main__":
    dashboard = NeutronMonitorDashboard()
    dashboard.start()