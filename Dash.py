import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import io
import base64
import json

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Helper function to parse uploaded file
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if 'ascii' in filename or 'txt' in filename:
        # Handle ASCII and TXT files
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t', skiprows=2)
    elif 'zip' in filename:
        # Handle ZIP files
        zip_file = zipfile.ZipFile(io.BytesIO(decoded))
        for file_name in zip_file.namelist():
            if file_name.endswith('.ascii') or file_name.endswith('.txt'):
                with zip_file.open(file_name) as f:
                    df = pd.read_csv(f, delimiter='\t', skiprows=2)
    
    clean_columns = df.columns
    return df.to_json(orient='split'), clean_columns

# Define layout for the Dash app
app.layout = dbc.Container(
    [
        html.H1("ASCII & TXT Data Viewer and Plotter (ZIP Support) with Heatmap", className="my-4"),

        # File uploader
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-file-info'),
        dcc.Store(id='stored-data'),  # Hidden storage for the dataframe

        # Chart configuration
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='xaxis-column', options=[], placeholder="Select X-axis")),
                dbc.Col(dcc.Dropdown(id='yaxis-column', options=[], placeholder="Select Y-axis (For Heatmap Only)", disabled=True))
            ]),
            dbc.Row([
                dbc.Col(dcc.Input(id='bucket-size-x', type='number', placeholder='Bucket Size for X-axis', value=10)),
                dbc.Col(dcc.Input(id='bucket-size-y', type='number', placeholder='Bucket Size for Y-axis', value=10, disabled=True))
            ]),
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='chart-type',
                    options=[
                        {'label': 'Line Chart', 'value': 'Line'},
                        {'label': 'Bar Chart', 'value': 'Bar'},
                        {'label': 'Scatter Plot', 'value': 'Scatter'},
                        {'label': 'Histogram', 'value': 'Histogram'},
                        {'label': 'Heatmap', 'value': 'Heatmap'}
                    ],
                    placeholder='Select Chart Type'
                )),
                dbc.Col(dbc.Button('Generate Chart', id='generate-chart', color='primary', className="btn-block"))
            ])
        ], className='my-3'),

        # Plot output
        dcc.Loading(
            id='loading-output',
            children=[dcc.Graph(id='chart-output')],
            type='default'
        )
    ],
    fluid=True
)

# Callback to store data and update file info
@app.callback(
    [Output('output-file-info', 'children'),
     Output('xaxis-column', 'options'),
     Output('xaxis-column', 'disabled'),
     Output('yaxis-column', 'disabled'),
     Output('stored-data', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_file_info(contents, filename):
    if contents is None:
        raise PreventUpdate
    data_json, clean_columns = parse_contents(contents, filename)
    msg = f"File {filename} successfully loaded!"
    return (
        msg,
        [{'label': col, 'value': col} for col in clean_columns],
        False,  # Enable X-axis dropdown
        False,  # Enable Y-axis dropdown (for Heatmap)
        data_json  # Store the data as JSON
    )

# Callback to generate chart based on user input
@app.callback(
    Output('chart-output', 'figure'),
    [Input('generate-chart', 'n_clicks')],
    [State('xaxis-column', 'value'),
     State('yaxis-column', 'value'),
     State('bucket-size-x', 'value'),
     State('bucket-size-y', 'value'),
     State('chart-type', 'value'),
     State('stored-data', 'data')]
)
def generate_chart(n_clicks, xaxis, yaxis, bucket_size_x, bucket_size_y, chart_type, stored_data):
    if n_clicks is None or stored_data is None:
        raise PreventUpdate
    
    # Convert the stored JSON data back to a dataframe
    df = pd.read_json(stored_data, orient='split')
    
    if chart_type == 'Line':
        fig = px.line(df, x=xaxis, y=yaxis, title=f"Line Chart ({xaxis} vs {yaxis})")
    elif chart_type == 'Bar':
        fig = px.bar(df, x=xaxis, y=yaxis, title=f"Bar Chart ({xaxis} vs {yaxis})")
    elif chart_type == 'Scatter':
        fig = px.scatter(df, x=xaxis, y=yaxis, title=f"Scatter Plot ({xaxis} vs {yaxis})")
    elif chart_type == 'Histogram':
        fig = create_histogram(df, xaxis, bucket_size_x)
    elif chart_type == 'Heatmap':
        fig = create_heatmap(df, xaxis, yaxis, bucket_size_x, bucket_size_y)
    
    return fig

# Function to create a histogram
def create_histogram(df, xaxis, bucket_size_x):
    df[xaxis] = pd.to_numeric(df[xaxis], errors='coerce')
    df.dropna(subset=[xaxis], inplace=True)
    
    bin_edges = np.arange(df[xaxis].min(), df[xaxis].max() + bucket_size_x, bucket_size_x)
    df['Binned'] = pd.cut(df[xaxis], bins=bin_edges, include_lowest=True)
    
    grouped = df.groupby('Binned').size().reset_index(name='Counts')
    total_count = grouped['Counts'].sum()
    grouped['Percentage'] = (grouped['Counts'] / total_count) * 100 if total_count > 0 else 0
    grouped['Binned'] = grouped['Binned'].astype(str)
    
    fig = px.bar(grouped, x='Binned', y='Percentage', title=f"Histogram of {xaxis} (Bucket Size: {bucket_size_x})")
    return fig

# Function to create a heatmap
def create_heatmap(df, xaxis, yaxis, bucket_size_x, bucket_size_y):
    df[xaxis] = pd.to_numeric(df[xaxis], errors='coerce')
    df[yaxis] = pd.to_numeric(df[yaxis], errors='coerce')
    df.dropna(subset=[xaxis, yaxis], inplace=True)
    
    # Create bins
    bin_edges_x = np.linspace(df[xaxis].min(), df[xaxis].max(), bucket_size_x + 1)
    bin_edges_y = np.linspace(df[yaxis].min(), df[yaxis].max(), bucket_size_y + 1)
    
    df['Binned_X'] = pd.cut(df[xaxis], bins=bin_edges_x)
    df['Binned_Y'] = pd.cut(df[yaxis], bins=bin_edges_y)
    
    # Create pivot table
    pivot_table = df.pivot_table(index='Binned_X', columns='Binned_Y', aggfunc='size', fill_value=0)
    pivot_table_percentage = pivot_table / pivot_table.sum().sum() * 100
    
    # Generate heatmap using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table_percentage, annot=True, fmt=".1f", cmap="coolwarm")
    ax.set_title(f"Heatmap of {xaxis} vs {yaxis}")
    
    return px.imshow(pivot_table_percentage)

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
