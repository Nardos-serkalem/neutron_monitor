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

# Constants
DATA_BASE_URL = "http://196.188.116.60/Min_Data/"
USERNAME = "user"
PASSWORD = "@Ssgi*123"
DAYS_TO_FETCH = 1
BUFFER_SIZE = 480  # 8 hours of data (60 mins * 8)
UPDATE_INTERVAL = 60000  # 60 seconds
PARAMETERS = ['Neutron_Count', 'Altitude', 'Temperature', 'Pressure']

class NeutronMonitorDashboard:
    def __init__(self):
        self.auth = HTTPBasicAuth(USERNAME, PASSWORD)
        self.data = self._load_initial_data()
        self.last_timestamp = self.data.index[-1] if not self.data.empty else datetime.now()

        # Buffers for each parameter
        self.buffers = {param: deque(maxlen=BUFFER_SIZE) for param in PARAMETERS}
        self.lines = {}

        # Fill buffers with initial data
        for _, row in self.data.iterrows():
            for param in PARAMETERS:
                self.buffers[param].append((row.name, row[param]))

        # Setup plots
        self.fig, self.axs = plt.subplots(len(PARAMETERS), 1, figsize=(12, 8))
        self._configure_axes()


def _load_initial_data(self):
    folder_path = "data"  # or wherever your data files are
    dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp')  # adjust as needed
            dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs).sort_index()
        numeric_df = combined_df.select_dtypes(include='number')
        resampled_df = numeric_df.resample('1T').mean()
        return resampled_df
    else:
        return pd.DataFrame()  # return empty if no data



    def _get_csv_filenames(self):
        response = requests.get(DATA_BASE_URL, auth=self.auth)
        file_list = re.findall(r'(\d{7}_min_Ethiopia\.csv)', response.text)
        return [DATA_BASE_URL + f for f in sorted(file_list)[-DAYS_TO_FETCH:]]

    def _process_csv(self, file_url):
        response = requests.get(file_url, auth=self.auth)
        if response.ok:
            df = pd.read_csv(StringIO(response.text), sep=r'\s+|,', header=None,
                             names=["Time", "Neutron_Count", "Altitude", "Temperature", "Pressure"])
            match = re.search(r'(\d{4})(\d{3})', file_url)
            base_date = datetime(int(match[1]), 1, 1) + timedelta(days=int(match[2]) - 1)
            df['DateTime'] = df['Time'].apply(lambda t: base_date +
                                              timedelta(hours=int(t[:2]), minutes=int(t[3:])))
            return df.set_index('DateTime').replace(0, np.nan)
        return pd.DataFrame()

    def _configure_axes(self):
        time_format = DateFormatter('%m-%d %H')
        for idx, (ax, param) in enumerate(zip(self.axs, PARAMETERS)):
            ax.xaxis.set_major_locator(HourLocator(interval=3))
            ax.xaxis.set_major_formatter(time_format)
            ax.xaxis.set_minor_locator(HourLocator())
            ax.grid(True, which='major', linestyle='-', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            ax.set_ylabel(self._get_y_label(param), rotation=0, labelpad=20)
            ax.set_xlim(datetime.now() - timedelta(hours=8), datetime.now())
            line, = ax.plot([], [], lw=1.5)
            self.lines[param] = line

    def _get_y_label(self, param):
        return {
            'Neutron_Count': 'Counts\n',
            'Altitude': 'm\n',
            'Temperature': '°C\n',
            'Pressure': 'hPa\n'
        }[param]

    def _update_data(self):
        """Fetch and append only new data points after the last timestamp."""
        new_data = self._load_initial_data()
        if not new_data.empty:
            new_data = new_data[new_data.index > self.last_timestamp]
            if not new_data.empty:
                self.last_timestamp = new_data.index[-1]
                for ts, row in new_data.iterrows():
                    for param in PARAMETERS:
                        self.buffers[param].append((ts, row[param]))

    def _update_plot(self, frame):
        self._update_data()
        for idx, param in enumerate(PARAMETERS):
            times = [t for t, v in self.buffers[param] if not pd.isna(v)]
            values = [v for t, v in self.buffers[param] if not pd.isna(v)]

            if times and values:
                self.lines[param].set_data(times, values)
                self.axs[idx].set_xlim(datetime.now() - timedelta(hours=8), datetime.now())
                self.axs[idx].set_ylim(min(values) * 0.98, max(values) * 1.02)

        return list(self.lines.values())

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
