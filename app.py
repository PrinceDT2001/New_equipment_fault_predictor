import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Set page configuration for a better look and feel
st.set_page_config(
    page_title="Equipment Anomaly Predictor",
    page_icon="⚙️",
    layout="centered",
)

# --- App Title and Description ---
st.title("⚙️ Equipment Anomaly Predictor")
st.markdown("### Predict equipment faults based on sensor readings")
st.markdown(
    """
This application uses a trained Isolation Forest model to predict if an equipment reading is anomalous.
Select the equipment location and type, and then enter the sensor data below to get a prediction.
"""
)

# --- Load the trained model and data ---
@st.cache_resource
def load_model_and_data():
    """
    Loads the trained Isolation Forest model and the data.
    """
    try:
        model = joblib.load("isolation_forest_model.joblib")
        df = pd.read_csv("equipment_anomaly_data.csv")
    except FileNotFoundError:
        st.error("Error: `isolation_forest_model.joblib` or `equipment_anomaly_data.csv` not found.")
        st.info("Training a new model on the provided CSV data. This may take a moment.")
        try:
            df = pd.read_csv("equipment_anomaly_data.csv")
            model = IsolationForest(random_state=42)
            # We'll train a basic model without the categorical data for now
            # as the current IsolationForest model doesn't handle strings directly.
            model.fit(df[['temperature', 'pressure', 'vibration', 'humidity']])
            # Save the new model
            joblib.dump(model, "isolation_forest_model.joblib")
        except FileNotFoundError:
            st.error("Error: `equipment_anomaly_data.csv` is missing. The app cannot run without the data file.")
            st.stop()
    return model, df

model, df = load_model_and_data()

# --- User Input Form ---
with st.form("anomaly_prediction_form"):
    st.subheader("Enter Equipment Details and Sensor Readings")
    
    # Define a list of locations and equipment types
    locations = ["Kasua", "Tema", "Wineba", "Osu", "Kumasi"]
    equipment_types = ["Turbine", "Pump", "Compressor", "Microplate Shaker"]

    # Add dropdown menus for location and equipment type
    location = st.selectbox("Equipment Location", options=locations)
    equipment_type = st.selectbox("Equipment Type", options=equipment_types)

    # Add number inputs for sensor readings
    temperature = st.number_input(
        "Temperature (°C)", min_value=0.0, max_value=100.0, value=float(df['temperature'].mean())
    )
    pressure = st.number_input(
        "Pressure (kPa)", min_value=0.0, max_value=100.0, value=float(df['pressure'].mean())
    )
    vibration = st.number_input(
        "Vibration (mm/s)", min_value=0.0, max_value=10.0, value=float(df['vibration'].mean())
    )
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=float(df['humidity'].mean())
    )

    # Every form must have a submit button.
    submit_button = st.form_submit_button(
        label="Predict Anomaly",
        use_container_width=True,
    )

# --- Prediction Logic ---
if submit_button:
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame(
        [[temperature, pressure, vibration, humidity]],
        columns=['temperature', 'pressure', 'vibration', 'humidity']
    )

    # Make the prediction
    prediction = model.predict(input_data)
    
    st.markdown("---")
    st.subheader("Prediction Result")

    # Display the result
    if prediction[0] == -1:
        st.error(f"⚠️ **Prediction for {equipment_type} at {location}:** Anomaly Detected")
        st.write("The entered sensor readings are outside the normal range for this equipment.")
    else:
        st.success(f"✅ **Prediction for {equipment_type} at {location}:** No Anomaly Detected")
        st.write("The entered sensor readings are within the normal operating range.")

# --- Footer ---
st.markdown("---")
st.info("Built with ❤️ using Streamlit")
