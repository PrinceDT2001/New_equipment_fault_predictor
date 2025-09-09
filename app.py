import streamlit as st
import pandas as pd
import numpy as np

# Set the page configuration
# This must be the first Streamlit command in your script.
st.set_page_config(
    page_title="Equipment Anomaly Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load and Prepare Data ---
@st.cache_data
def load_data(file_path):
    """
    Loads the equipment anomaly data from a CSV file.
    st.cache_data ensures the data is only loaded once.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.stop()
        
df = load_data("equipment_anomaly_data.csv")

# Ensure required columns exist
required_columns = ['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location', 'faulty']
if not all(col in df.columns for col in required_columns):
    st.error("The CSV file is missing one or more required columns.")
    st.stop()

# Convert the 'faulty' column to boolean
df['faulty'] = df['faulty'].astype(bool)

# --- Sidebar Filters ---
with st.sidebar:
    st.header("⚙️ Filter Data")
    
    # Filter by equipment type
    equipment_types = df['equipment'].unique()
    selected_equipment = st.selectbox(
        "Select Equipment Type",
        options=equipment_types,
        index=0
    )
    
    # Filter by location
    locations = df['location'].unique()
    selected_location = st.multiselect(
        "Select Location(s)",
        options=locations,
        default=locations
    )

    # Filter by temperature
    temp_range = st.slider(
        "Temperature Range (°C)",
        min_value=float(df['temperature'].min()),
        max_value=float(df['temperature'].max()),
        value=(float(df['temperature'].min()), float(df['temperature'].max()))
    )
    
# Apply filters to the main dataframe
filtered_df = df[
    (df['equipment'] == selected_equipment) &
    (df['location'].isin(selected_location)) &
    (df['temperature'] >= temp_range[0]) &
    (df['temperature'] <= temp_range[1])
]

# --- Main Page Layout ---
st.title("Equipment Anomaly Detection Dashboard")
st.markdown("This dashboard provides an overview of equipment sensor data and potential anomalies.")

# Use st.columns to create a flexible, horizontal layout
col1, col2, col3 = st.columns(3)

# Display key metrics using st.metric
with col1:
    total_data_points = len(filtered_df)
    st.metric(label="Total Data Points", value=f"{total_data_points:,}")

with col2:
    faulty_count = filtered_df[filtered_df['faulty']].shape[0]
    st.metric(label="Faulty Equipment", value=f"{faulty_count:,}")

with col3:
    fault_rate = (faulty_count / total_data_points) * 100 if total_data_points > 0 else 0
    st.metric(label="Fault Rate", value=f"{fault_rate:.2f}%")

st.divider()

# --- Dataframe Display ---
st.subheader("Raw Data")
st.write("Displaying filtered data from the CSV file.")
st.dataframe(filtered_df, use_container_width=True)

st.divider()

# --- Data Visualization ---
st.subheader("Data Visualizations")

# Create two columns for the charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("Temperature and Pressure Trends")
    st.line_chart(filtered_df[['temperature', 'pressure']])

with chart_col2:
    st.write("Vibration Distribution")
    st.bar_chart(filtered_df['vibration'])

# You can also add other components like a map
st.subheader("Equipment Locations")
# Note: For a map, you'd typically need latitude and longitude data.
# This is just a placeholder to show the component.
# This component expects a DataFrame with columns named 'latitude' and 'longitude'
# st.map(filtered_df)
