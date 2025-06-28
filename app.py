import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
from utils import engineer_features

def load_model():
    """Load the pre-trained model."""
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    st.set_page_config(page_title="TripFare Prediction", page_icon="🚗")
    st.title('🚗 TripFare: Urban Taxi Fare Prediction')
    
    # Display EDA visualizations
    st.header('Exploratory Data Analysis')
    for img in ['fare_vs_distance.png', 'fare_distribution.png', 'fare_by_hour.png', 'fare_by_day.png']:
        try:
            st.image(img, caption=img.split('.')[0].replace('_', ' ').title())
        except FileNotFoundError:
            st.write(f"Visualization {img} not found. Run main.py first to generate visualizations.")
    
    # User input interface
    st.header('Predict Your Taxi Fare')
    
    col1, col2 = st.columns(2)
    
    with col1:
        pickup_lat = st.number_input('Pickup Latitude', min_value=40.5, max_value=41.0, value=40.7128, format="%.6f")
        pickup_lon = st.number_input('Pickup Longitude', min_value=-74.5, max_value=-73.5, value=-74.0060, format="%.6f")
        dropoff_lat = st.number_input('Dropoff Latitude', min_value=40.5, max_value=41.0, value=40.7128, format="%.6f")
        dropoff_lon = st.number_input('Dropoff Longitude', min_value=-74.5, max_value=-73.5, value=-74.0060, format="%.6f")
    
    with col2:
        passenger_count = st.number_input('Passenger Count', min_value=1, max_value=6, value=1)
        pickup_time = st.time_input('Pickup Time', value=datetime.now().time())
        pickup_date = st.date_input('Pickup Date', value=datetime.now())
        rate_code = st.selectbox('Rate Code', [1, 2, 3, 4, 5], help="1=Standard, 2=JFK, 3=Newark, 4=Nassau, 5=Negotiated")
        payment_type = st.selectbox('Payment Type', [1, 2, 3, 4], help="1=Credit card, 2=Cash, 3=No charge, 4=Dispute")
    
    if st.button('Predict Fare'):
        # Create input dataframe
        pickup_datetime = pd.to_datetime(f"{pickup_date} {pickup_time}")
        # Assume a default trip duration of 15 minutes
        dropoff_datetime = pickup_datetime + timedelta(minutes=15)
        
        input_data = pd.DataFrame({
            'pickup_latitude': [pickup_lat],
            'pickup_longitude': [pickup_lon],
            'dropoff_latitude': [dropoff_lat],
            'dropoff_longitude': [dropoff_lon],
            'passenger_count': [passenger_count],
            'RatecodeID': [rate_code],
            'payment_type': [payment_type],
            'tpep_pickup_datetime': [pickup_datetime],
            'tpep_dropoff_datetime': [dropoff_datetime],
            'fare_amount': [0.0]  # Dummy value, as it's required for fare_per_mile calculation
        })
        
        # Engineer features
        try:
            input_data = engineer_features(input_data)
        except ValueError as e:
            st.error(f"Error in feature engineering: {e}")
            return
        
        # Load and predict
        try:
            model = load_model()
            prediction = model.predict(input_data)
            st.success(f'Predicted Fare: ${prediction[0]:.2f}')
        except FileNotFoundError:
            st.error("Model file not found. Please run main.py to train the model first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

if __name__ == '__main__':
    main()