from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import zipfile

app = Flask(__name__, template_folder='templates')

# Function to load and process the data from ASCII or TXT file
def load_data(file_content: str):
    df = pd.read_csv(StringIO(file_content), delimiter='\t', skiprows=2)
    df.drop(index=0, inplace=True)  # Drop unnecessary rows
    df.reset_index(drop=True, inplace=True)  # Reset index
    return df

# Function to handle ZIP files and extract the first valid ASCII or TXT file
def extract_first_ascii_txt_from_zip(zip_file):
    with zipfile.ZipFile(zip_file) as z:
        for filename in z.namelist():
            if filename.endswith('.ascii') or filename.endswith('.txt'):
                with z.open(filename) as f:
                    return f.read().decode('utf-8')
    return None

# Function to create charts based on type
def create_chart(df, chart_type, x_axis, y_axis, bucket_size=None):
    if chart_type == 'Line':
        return px.line(df, x=x_axis, y=y_axis, title=f"Line Chart ({x_axis} vs {y_axis})")
    elif chart_type == 'Bar':
        return px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart ({x_axis} vs {y_axis})")
    elif chart_type == 'Scatter':
        return px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Chart ({x_axis} vs {y_axis})")
    elif chart_type == 'Histogram':
        return create_custom_histogram(df, x_axis, y_axis, bucket_size)
    elif chart_type == 'Heatmap':
        return create_heatmap(df, x_axis, y_axis)

# Custom histogram function
def create_custom_histogram(df, x_axis, y_axis, bucket_size):
    df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
    df.dropna(subset=[x_axis, y_axis], inplace=True)
    
    bin_edges = np.arange(df[x_axis].min(), df[x_axis].max() + bucket_size, bucket_size)
    df['Binned'] = pd.cut(df[x_axis], bins=bin_edges, include_lowest=True)
    
    grouped = df.groupby('Binned', observed=False)[y_axis].sum().reset_index()
    total_sum = grouped[y_axis].sum()
    grouped['Percentage'] = (grouped[y_axis] / total_sum) * 100 if total_sum > 0 else 0
    
    grouped['Binned'] = grouped['Binned'].astype(str)
    fig = px.bar(grouped, x='Binned', y='Percentage', title=f"Histogram of {y_axis} over {x_axis} (Bucket Size: {bucket_size})")
    
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside', textangle=0)
    return fig

# Heatmap function
def create_heatmap(df, main_var, var1, var2):
    df[main_var] = pd.to_numeric(df[main_var], errors='coerce')
    df[var1] = pd.to_numeric(df[var1], errors='coerce')
    df[var2] = pd.to_numeric(df[var2], errors='coerce')

    df = df.dropna(subset=[main_var, var1, var2])
    heatmap_data = df.pivot_table(values=main_var, index=var1, columns=var2, aggfunc='mean')
    return px.imshow(heatmap_data, title="Heatmap of Main Variable vs Two Variables")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    file_content = None

    # If the file is a ZIP file, extract the first ASCII or TXT file
    if file and file.filename.endswith('.zip'):
        file_content = extract_first_ascii_txt_from_zip(file)

    # If it's a regular ASCII or TXT file
    elif file and (file.filename.endswith('.ascii') or file.filename.endswith('.txt')):
        file_content = file.read().decode('utf-8')

    if file_content:
        df = load_data(file_content)

        # Get form data for chart generation
        x_axis = request.form['x_axis']
        y_axis = request.form['y_axis']
        chart_type = request.form['chart_type']
        bucket_size = int(request.form.get('bucket_size', 200))

        # Generate the selected chart
        fig = create_chart(df, chart_type, x_axis, y_axis, bucket_size)
        chart_html = fig.to_html(full_html=False)

        return render_template('chart.html', chart_html=chart_html)

    return "Please upload a valid ASCII, TXT or ZIP file.", 400

@app.route('/get_columns', methods=['POST'])
def get_columns():
    file = request.files['file']
    file_content = None

    # Handle ZIP file extraction
    if file and file.filename.endswith('.zip'):
        file_content = extract_first_ascii_txt_from_zip(file)

    elif file and (file.filename.endswith('.ascii') or file.filename.endswith('.txt')):
        file_content = file.read().decode('utf-8')

    if file_content:
        df = load_data(file_content)
        columns = df.columns.tolist()
        return jsonify(columns)

    return jsonify([]), 400

if __name__ == '__main__':
    app.run(debug=True)
