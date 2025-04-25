import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
import requests
from requests.auth import HTTPBasicAuth
from io import StringIO

# URL and login
data_base_url = "http://196.188.116.60/Min_Data/"
username = "user"
password = "@Ssgi*123"

# How many days to fetch
days_to_fetch = 3

# Use today's date in UTC
today = datetime.now(timezone.utc).date()

# Build a list of file names for the last 3 days
file_list = []
for i in range(days_to_fetch):
    date = today - timedelta(days=i)
    file_name = f"{date.strftime('%Y-%m-%d')}.csv"
    file_list.append(file_name)

# Fetch and combine data
df_all = pd.DataFrame()
for file_name in file_list:
    file_url = data_base_url + file_name
    try:
        response = requests.get(file_url, auth=HTTPBasicAuth(username, password))
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, index_col='Time', parse_dates=True)
        df_all = pd.concat([df_all, df])
        print(f"Fetched {file_name}")
    except Exception as e:
        print(f"Could not fetch {file_name}: {e}")

# Check if we have any data
if df_all.empty:
    print("No data fetched. Exiting...")
    exit()

# Filter time range
start_time = today - timedelta(days=days_to_fetch - 1)
df_all = df_all[(df_all.index >= start_time)]

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Gray background
fig.patch.set_facecolor('gray')
for ax in axs:
    ax.set_facecolor('gray')

# Plot Neutron Count
axs[0].scatter(df_all.index, df_all['Neutron'], color='blue', s=10)
axs[0].set_ylabel('Counts')
axs[0].set_title('Neutron Count')

# Plot Temperature
axs[1].scatter(df_all.index, df_all['Temperature'], color='orange', s=10)
axs[1].set_ylabel('°C')
axs[1].set_title('Temperature')

# Plot Pressure
axs[2].scatter(df_all.index, df_all['Pressure'], color='purple', s=10)
axs[2].set_ylabel('hPa')
axs[2].set_title('Pressure')

# Format x-axis: every 6 hours
axs[2].xaxis.set_major_locator(mdates.HourLocator(interval=6))
axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

# Set common x-axis label
axs[2].set_xlabel('Time')

# Improve layout
plt.tight_layout()

# Show plot
plt.show()
