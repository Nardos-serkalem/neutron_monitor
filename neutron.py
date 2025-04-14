import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from io import StringIO
import re
from collections import deque

DATA_BASE_URL = "http://196.188.116.60/Min_Data/"
USERNAME = "user"
PASSWORD = "@Ssgi*123"
DAYS_TO_FETCH = 3
BUFFER_SIZE = 840  
Z_THRESHOLD = 3
UPDATE_INTERVAL = 120000 

class NeutronMonitorVisualizer:
    def __init__(self):
        self.auth = HTTPBasicAuth(USERNAME, PASSWORD)
        self.colors = ['blue', 'green', 'orange', 'purple']
        self.raw_data = self._load_initial_data()
        self.current_index = 0
        

        self.fig, self.axs = plt.subplots(4, 1, figsize=(15, 10))
        self._setup_plots()
        
        # Data buffers with outlier removal
        self.buffers = {
            col: {
                'raw': deque(maxlen=BUFFER_SIZE),
                'processed': deque(maxlen=BUFFER_SIZE),
                'window': deque(maxlen=60)  # 1-hour window for stats
            } for col in ['Neutron_Count', 'Altitude', 'Temperature', 'Pressure']
        }

    def _load_initial_data(self):
        file_urls = self._get_csv_filenames()
        dfs = [self._process_csv(url) for url in file_urls]
        df_all = pd.concat(dfs).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        return df_all.replace(0, np.nan)

    def _get_csv_filenames(self):
        response = requests.get(DATA_BASE_URL, auth=self.auth)
        file_list = re.findall(r'(\d{7}_min_Ethiopia\.csv)', response.text)
        return [DATA_BASE_URL + f for f in sorted(file_list)[-DAYS_TO_FETCH:]]

    def _process_csv(self, file_url):
        response = requests.get(file_url, auth=self.auth)
        if response.status_code != 200:
            return pd.DataFrame()
        
        df = pd.read_csv(StringIO(response.text), sep=r'\s+|,', engine='python', header=None,
                         names=["Time", "Neutron_Count", "Altitude", "Temperature", "Pressure"])
        
        match = re.search(r'(\d{4})(\d{3})_min_Ethiopia\.csv', file_url)
        year, doy = int(match.group(1)), int(match.group(2))
        base_date = datetime(year, 1, 1) + timedelta(days=doy-1)
        
        df['DateTime'] = df['Time'].apply(
            lambda t: base_date + timedelta(hours=int(t[:2]), minutes=int(t[3:5])))
        return df.set_index('DateTime').drop(columns=['Time'])

    def _setup_plots(self):
        titles = ['Neutron Count', 'Altitude', 'Temperature', 'Pressure']
        y_labels = ['Counts', 'm', '°C', 'hPa']
        
        self.lines = {
            'Neutron_Count': {'raw': None, 'processed': None},
            'Altitude': {'raw': None, 'processed': None},
            'Temperature': {'raw': None, 'processed': None},
            'Pressure': {'raw': None, 'processed': None}
        }
        
        for idx, (ax, title, ylabel, color) in enumerate(zip(self.axs, titles, y_labels, self.colors)):
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.set_xlim(datetime.now() - timedelta(hours=24), datetime.now())
            
            # Initialize lines
            self.lines[title.replace(' ', '_')]['raw'], = ax.plot([], [], color=color, alpha=0.2, lw=0.5)
            self.lines[title.replace(' ', '_')]['processed'], = ax.plot([], [], color=color, lw=1)

    def _remove_outliers(self, col, new_value):
        buffer = self.buffers[col]
        buffer['window'].append(new_value)
        
        if len(buffer['window']) >= 60:
            window = np.array(buffer['window'])
            mean = np.nanmean(window)
            std = np.nanstd(window)
            
            if std == 0 or np.isnan(std):
                return new_value
            z = (new_value - mean) / std
            return mean if abs(z) > Z_THRESHOLD else new_value
        return new_value

    def _update_plot(self, frame):
        if self.current_index >= len(self.raw_data):
            self.current_index = 0  # Loop data
            
        row = self.raw_data.iloc[self.current_index]
        timestamp = self.raw_data.index[self.current_index]
        
        artists = []
        for col in ['Neutron_Count', 'Altitude', 'Temperature', 'Pressure']:
            value = row[col]
            if pd.isna(value):
                continue
                
            # Process value
            cleaned = self._remove_outliers(col, value)
            
            # Update buffers
            self.buffers[col]['raw'].append((timestamp, value))
            self.buffers[col]['processed'].append((timestamp, cleaned))
            
            # Update plot data
            ax = self.axs[['Neutron_Count', 'Altitude', 'Temperature', 'Pressure'].index(col)]
            x_proc = [t for t, v in self.buffers[col]['processed']]
            y_proc = [v for t, v in self.buffers[col]['processed']]
            x_raw = [t for t, v in self.buffers[col]['raw']]
            y_raw = [v for t, v in self.buffers[col]['raw']]
            
            self.lines[col]['processed'].set_data(x_proc, y_proc)
            self.lines[col]['raw'].set_data(x_raw, y_raw)
            
            # Dynamic Y-axis scaling
            valid_vals = [v for v in y_proc if not np.isnan(v)]
            if valid_vals:
                ymin = np.nanmin(valid_vals)
                ymax = np.nanmax(valid_vals)
                padding = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
                ax.set_ylim(ymin - padding, ymax + padding)
            
            artists.extend([self.lines[col]['processed'], self.lines[col]['raw']])
        
        self.current_index += 1
        return artists

    def start(self):
        ani = FuncAnimation(
            self.fig, 
            self._update_plot, 
            interval=UPDATE_INTERVAL, 
            blit=True,
            cache_frame_data=False,
            save_count=BUFFER_SIZE
        )
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualizer = NeutronMonitorVisualizer()
    visualizer.start()