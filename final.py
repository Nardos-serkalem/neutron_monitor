import dash
from dash import dcc, html, Input, Output
import os
import time
import subprocess
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from io import StringIO
import re
from datetime import datetime, timedelta

app = dash.Dash(__name__)
server = app.server

DARK_THEME = {
    'background': '#111111',
    'text': '#e6e6e6',
    'card_bg': '#222222',
    'grid_line': '#404040'
}
UPDATE_INTERVAL = 600  # 10 min

# Dashboard layout
app.layout = html.Div([
    dcc.Interval(id='interval', interval=UPDATE_INTERVAL * 1000, n_intervals=0),
    
    html.Div([
        html.H1("Neutron Monitoring Dashboard", 
                style={'color': DARK_THEME['text'], 'padding': '20px', 'textAlign': 'center'}),
        
        html.Div([
            html.Div([   
                html.Div([ 
                    html.H3("Neutron Monitor Data", style={'color': DARK_THEME['text']}), 
                    html.Img(id='neutron-plot', style={'height': '45vh', 'width': '100%'})
                ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '15px', 'borderRadius': '10px'}), 

                html.Div([ 
                    html.H3("GCR Radiation Levels", style={'color': DARK_THEME['text']}), 
                    html.Img(id='gcr-plot', style={'height': '45vh', 'width': '100%'})
                ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '15px', 'borderRadius': '10px'})
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}), 

            # Second Row
            html.Div([ 
                html.Div([ 
                    html.H3("Radiation Dose Rates", style={'color': DARK_THEME['text']}), 
                    html.Img(id='dose-gif', style={'height': '45vh', 'width': '100%'}) 
                ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '15px', 'borderRadius': '10px'}), 

                html.Div([ 
                    html.H3("GOES X-ray Flux", style={'color': DARK_THEME['text']}),  
                    html.Img(id='goes-plot', style={'height': '45vh', 'width': '100%'})  
                ], style={'backgroundColor': DARK_THEME['card_bg'], 'padding': '15px', 'borderRadius': '10px'}) 
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}) 
        ], style={'padding': '20px'}) 
    ], style={'backgroundColor': DARK_THEME['background'], 'height': '100vh'}) 
])

# Update plots function that calls the neutron script and updates images
@app.callback(
    [Output('neutron-plot', 'src'),
     Output('gcr-plot', 'src'),
     Output('dose-gif', 'src'),
     Output('goes-plot', 'src')],
    [Input('interval', 'n_intervals')]
)
def update_plots(n):
    # Run neutron.py script to generate the plot
    subprocess.run(['python3', 'neutron.py'], check=True)

    ts = int(time.time())

    return [
        f"neutron_plot.png?{ts}",  # Plot generated by neutron.py
        f"https://wasavies.nict.go.jp/slides/WorldDose2/GCR.12.00.rel.png?{ts}",
        f"https://avidos.seibersdorf-laboratories.at/V2-0/CURRENT_AVIDOS.eu_c_SeibersdorfLaborGmbH.gif?{ts}",
        f"https://wasavies.nict.go.jp/slides/Summary2/latest_graphgoes.png?{ts}"
    ]

# Neutron script that fetches data and generates the plot
def generate_neutron_plot():
    data_base_url = "http://196.188.116.60/Min_Data/"
    username = "user"
    password = "@Ssgi*123"
    days_to_fetch = 3

    def get_csv_filenames_from_url(base_url, auth):
        response = requests.get(base_url, auth=auth)
        file_list = re.findall(r'(\d{7}_min_Ethiopia\.csv)', response.text)
        file_list = sorted(file_list)[-days_to_fetch:]
        return [base_url + fname for fname in file_list]

    def read_and_process_csv_from_url(file_url, auth):
        response = requests.get(file_url, auth=auth)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), sep=r'\s+|,', engine='python', header=None,
                             names=["Time", "Neutron_Count", "Altitude", "Temperature", "Pressure"])
            
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
            df = df.replace(0, np.nan)
            return df
        else:
            print(f"Failed to download: {file_url} (Status: {response.status_code})")
            return pd.DataFrame()

    auth = HTTPBasicAuth(username, password)
    file_urls = get_csv_filenames_from_url(data_base_url, auth)
    dfs = [read_and_process_csv_from_url(url, auth) for url in file_urls]
    df_all = pd.concat(dfs).sort_index()

    df_all = df_all[~df_all.index.duplicated(keep='first')]

    # Plotting the data
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), facecolor='#111111')
    axs[0].scatter(df_all.index, df_all['Neutron_Count'], s=5, color='cyan')
    axs[0].set_title('Neutron Count', color='white')
    axs[0].set_ylabel('Counts', color='white')
    axs[0].set_facecolor('#222222')
    axs[0].grid(True, linestyle='--', color='#404040')
    axs[0].tick_params(axis='x', colors='white')
    axs[0].tick_params(axis='y', colors='white')

    axs[1].scatter(df_all.index, df_all['Temperature'], s=5, color='orange')
    axs[1].set_title('Temperature', color='white')
    axs[1].set_ylabel('°C', color='white')
    axs[1].set_facecolor('#222222')
    axs[1].grid(True, linestyle='--', color='#404040')
    axs[1].tick_params(axis='x', colors='white')
    axs[1].tick_params(axis='y', colors='white')

    axs[2].scatter(df_all.index, df_all['Pressure'], s=5, color='purple')
    axs[2].set_title('Pressure', color='white')
    axs[2].set_ylabel('hPa', color='white')
    axs[2].set_xlabel('Time (UTC)', color='white')
    axs[2].set_facecolor('#222222')
    axs[2].grid(True, linestyle='--', color='#404040')
    axs[2].tick_params(axis='x', colors='white')
    axs[2].tick_params(axis='y', colors='white')

    fig.autofmt_xdate()
    plt.tight_layout()

    # Save the figure
    output_path = "neutron_plot.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)

    print(f"Saved neutron plot to {output_path}")

if __name__ == '__main__':
    app.run(debug=True, port=8050)
