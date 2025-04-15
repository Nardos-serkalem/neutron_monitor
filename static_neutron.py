import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime, timedelta
import os
import requests
from requests.auth import HTTPBasicAuth

class NeutronDataAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.df = self._load_and_preprocess()
        self._clean_data()
        
    def _parse_datetime(self):
        """Extract date from filename (format: YYYYDDD_min_Ethiopia.csv)"""
        match = re.search(r'(\d{4})(\d{3})', self.filename)
        year, doy = int(match.group(1)), int(match.group(2))
        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        return base_date

    def _load_and_preprocess(self):
        """Load CSV and create datetime index"""
        # Load data
        df = pd.read_csv(self.filename, sep=r'\s+|,', engine='python', header=None,
                        names=["Time", "Neutron_Count", "Altitude", "Temperature", "Pressure"])
        
        # Parse datetime
        base_date = self._parse_datetime()
        df['DateTime'] = df['Time'].apply(
            lambda t: base_date + timedelta(hours=int(t[:2]), minutes=int(t[3:])))
        
        # Set index and clean
        df = df.set_index('DateTime').drop(columns=['Time', 'Altitude'])
        return df.asfreq('1T')  # Ensure regular frequency

    def _zscore_outlier_removal(self, series, window=60, threshold=3):
        """Replace outliers using rolling Z-score"""
        roll_mean = series.rolling(window=window, min_periods=1).mean()
        roll_std = series.rolling(window=window, min_periods=1).std().replace(0, 1e-6)
        zscore = np.abs((series - roll_mean) / roll_std)
        return series.where(zscore < threshold, roll_mean)

    def _clean_data(self):
        """Process data with outlier removal and gap filling"""
        # Remove outliers
        self.df['Neutron_Clean'] = self._zscore_outlier_removal(self.df['Neutron_Count'])
        self.df['Temp_Clean'] = self._zscore_outlier_removal(self.df['Temperature'])
        self.df['Pressure_Clean'] = self._zscore_outlier_removal(self.df['Pressure'])
        
        # Fill gaps using time-based interpolation
        self.df = self.df.interpolate(method='time')

    def plot_results(self):
        """Visualize raw vs cleaned data using dot plots"""
        fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Common plot parameters
        dot_size = 8
        alpha_raw = 0.3
        alpha_clean = 0.7
        
        # Neutron Count plot
        axs[0].scatter(self.df.index, self.df['Neutron_Count'], 
                      s=dot_size, color='blue', alpha=alpha_raw, label='Raw')
        axs[0].scatter(self.df.index, self.df['Neutron_Clean'], 
                      s=dot_size, color='blue', alpha=alpha_clean, label='Cleaned')
        axs[0].set_ylabel('Neutron Count\n(CPS)', rotation=0, ha='right')
        
        # Temperature plot
        axs[1].scatter(self.df.index, self.df['Temperature'], 
                      s=dot_size, color='orange', alpha=alpha_raw)
        axs[1].scatter(self.df.index, self.df['Temp_Clean'], 
                      s=dot_size, color='orange', alpha=alpha_clean)
        axs[1].set_ylabel('Temperature\n(°C)', rotation=0, ha='right')
        
        # Pressure plot
        axs[2].scatter(self.df.index, self.df['Pressure'], 
                      s=dot_size, color='purple', alpha=alpha_raw)
        axs[2].scatter(self.df.index, self.df['Pressure_Clean'], 
                      s=dot_size, color='purple', alpha=alpha_clean)
        axs[2].set_ylabel('Pressure\n(hPa)', rotation=0, ha='right')
        
        # Formatting
        for ax in axs:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='upper right')
            
        plt.xlabel('DateTime')
        plt.tight_layout()
        plt.show()

# ---- EXECUTION ----
if __name__ == "__main__":
    analyzer = NeutronDataAnalyzer("2025104_min_Ethiopia.csv")
    analyzer.plot_results()