import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
import tempfile
import base64  # For encoding the image

# Set Streamlit page configuration to expand width
st.set_page_config(layout="wide")

# Function to display the logo in the top-right corner
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    return b64_string

# Path to your local image
logo_path = r"C:\Users\ar705189\Documents\Apurva\Plotting_Application\koel_logo.png"  # Replace with your actual path
logo_base64 = get_base64_image(logo_path)

# Embed the logo image in HTML with base64 encoding
st.markdown(
    f"""
    <style>
        .logo-container {{
            position: absolute;
            top: 10px;
            right: 10px;
        }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" alt="Company Logo" width="100">
    </div>
    """,
    unsafe_allow_html=True
)

# Function to load and downcast data with Pandas, handling ZIP files by extracting them to a temporary directory
def load_data_with_pandas(file, file_type, rows_to_read=None):
    try:
        # Handle ASCII or TXT file directly
        if file_type == 'ascii':
            df = pd.read_csv(file, delimiter='\t', skiprows=2, nrows=rows_to_read)
        elif file_type == 'txt':
            df = pd.read_csv(file, delimiter='\t', nrows=rows_to_read)
        elif file_type == 'zip':
            with zipfile.ZipFile(file) as z:
                for filename in z.namelist():
                    if filename.endswith('.ascii') or filename.endswith('.txt'):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmpfile:
                            extracted_file_path = tmpfile.name
                            with z.open(filename) as f:
                                tmpfile.write(f.read())
                        df = pd.read_csv(extracted_file_path, delimiter='\t', skiprows=2, nrows=rows_to_read)
                        break

        # Downcast numeric columns to reduce memory usage
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float' if df[col].dtype == 'float64' else 'integer')

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to create charts based on user input
def create_chart(df, chart_type, x_axis, y_axis, x_bucket_size=None, y_bucket_size=None):
    if chart_type == 'Line':
        return px.line(df, x=x_axis, y=y_axis, title=f"Line Chart ({x_axis} vs {y_axis})")
    elif chart_type == 'Bar':
        return px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart ({x_axis} vs {y_axis})")
    elif chart_type == 'Scatter':
        return px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot ({x_axis} vs {y_axis})")
    elif chart_type == 'Histogram':
        return create_custom_histogram(df, x_axis, int(x_bucket_size))
    elif chart_type == 'Heatmap':
        return create_heatmap(df, x_axis, y_axis, int(x_bucket_size), int(y_bucket_size))

# Custom histogram function
def create_custom_histogram(df, x_axis, bucket_size):
    df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
    df.dropna(subset=[x_axis], inplace=True)
    
    bin_edges = pd.cut(df[x_axis], bins=bucket_size)
    df['Binned'] = bin_edges.astype(str)  # Convert intervals to strings

    grouped = df.groupby('Binned').size().reset_index(name='Counts')
    total_count = grouped['Counts'].sum()
    grouped['Percentage'] = (grouped['Counts'] / total_count) * 100 if total_count > 0 else 0

    fig = px.bar(grouped, x='Binned', y='Percentage', title=f"Histogram of {x_axis} (Bucket Size: {bucket_size})")
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside', textangle=0)
    return fig

# Heatmap function with both x and y bucket sizes
def create_heatmap(df, x_axis, y_axis, x_bucket_size, y_bucket_size):
    df[x_axis] = pd.to_numeric(df[x_axis], errors='coerce')
    df[y_axis] = pd.to_numeric(df[y_axis], errors='coerce')
    df.dropna(subset=[x_axis, y_axis], inplace=True)

    # Bin both x and y axes and convert to strings
    df['Binned_X'] = pd.cut(df[x_axis], bins=x_bucket_size).astype(str)
    df['Binned_Y'] = pd.cut(df[y_axis], bins=y_bucket_size).astype(str)

    # Create a pivot table for the heatmap
    pivot_table = df.pivot_table(index='Binned_Y', columns='Binned_X', aggfunc='size', fill_value=0)
    pivot_table_percentage = (pivot_table / pivot_table.sum().sum()) * 100

    # Plot the heatmap
    fig = px.imshow(
        pivot_table_percentage, 
        labels=dict(x=x_axis, y=y_axis, color="Percentage"),
        title=f"Heatmap of {x_axis} vs {y_axis} (Bucket Sizes: {x_bucket_size}, {y_bucket_size})"
    )
    fig.update_xaxes(side="top")
    fig.update_traces(texttemplate='%{z:.1f}%', textfont_size=12)  # Display percentage values within cells
    return fig

# Streamlit UI for file upload and chart generation
st.title("CPE Voluminous Data Analysis")

uploaded_file = st.file_uploader("Upload an ASCII, TXT, or ZIP file", type=['ascii', 'txt', 'zip'])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.ascii'):
        file_type = 'ascii'
    elif uploaded_file.name.endswith('.txt'):
        file_type = 'txt'
    elif uploaded_file.name.endswith('.zip'):
        file_type = 'zip'

    # Load the full dataset using Pandas
    df = load_data_with_pandas(uploaded_file, file_type)

    if df is not None:
        st.success("File loaded successfully!")
        
        # Display only memory usage
        st.write(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")
        
        # Initialize session state for storing multiple chart configurations and a dummy variable for rerun trigger
        if 'chart_configs' not in st.session_state:
            st.session_state.chart_configs = []
        if 'rerun_trigger' not in st.session_state:
            st.session_state.rerun_trigger = 0  # Dummy variable to trigger rerun

        # Form to configure and add a new chart
        with st.form("chart_form"):
            st.subheader("Configure a New Chart")

            clean_columns = df.columns.tolist()
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
            with col1:
                x_axis = st.selectbox("Select X-axis", clean_columns, key="x_axis")
            with col2:
                y_axis = st.selectbox("Select Y-axis", clean_columns, index=1, key="y_axis")
            with col3:
                chart_type = st.selectbox("Select Chart Type", ['Line', 'Bar', 'Scatter', 'Histogram', 'Heatmap'], key="chart_type")
            with col4:
                x_bucket_size = st.number_input("X Bucket Size", min_value=1, value=10, step=1, key="x_bucket_size")
            with col5:
                y_bucket_size = st.number_input("Y Bucket Size", min_value=1, value=10, step=1, key="y_bucket_size")
            
            add_chart = st.form_submit_button("Add Chart")
            
            if add_chart:
                # Add new chart configuration to session state
                st.session_state.chart_configs.append({
                    "chart_type": chart_type,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "x_bucket_size": x_bucket_size,
                    "y_bucket_size": y_bucket_size
                })
                st.session_state.rerun_trigger += 1  # Trigger rerun

        # Display each chart with a remove button
        st.subheader("Your Charts")
        for idx, config in enumerate(st.session_state.chart_configs):
            fig = create_chart(
                df,
                config["chart_type"],
                config["x_axis"],
                config["y_axis"],
                config["x_bucket_size"],
                config["y_bucket_size"]
            )
            
            # Display chart and remove button in a single row
            col_chart, col_button = st.columns([10, 1])
            with col_chart:
                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{idx}")
            with col_button:
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.chart_configs.pop(idx)
                    st.session_state.rerun_trigger += 1  # Trigger rerun to update UI

    else:
        st.error("Unable to load the file.")