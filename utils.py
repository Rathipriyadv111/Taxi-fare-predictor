import pandas as pd
import numpy as np
from haversine import haversine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def engineer_features(df):
    """Perform feature engineering on the dataset."""
    # Validate required columns
    required_columns = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 
                        'dropoff_longitude', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert datetime columns
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

    # Localize to UTC before converting to US/Eastern
    try:
        df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        df['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    except TypeError as e:
        print(f"Error handling timezone: {e}. Skipping timezone conversion.")
        # If timezone conversion fails, proceed without it (optional fallback)
    
    # Calculate trip distance using Haversine formula
    df['trip_distance'] = df.apply(lambda row: haversine(
        (row['pickup_latitude'], row['pickup_longitude']),
        (row['dropoff_latitude'], row['dropoff_longitude'])
    ), axis=1)
    
    # Extract time-based features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
    df['is_weekend'] = df['pickup_day'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] < 6)).astype(int)
    df['is_rush_hour'] = (df['pickup_hour'].isin([7, 8, 9, 16, 17, 18])).astype(int)
    
    # Calculate trip duration in minutes
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Calculate fare per mile
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance'].replace(0, np.nan)
    
    return df

def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    # Handle missing values
    df = df.dropna(subset=['total_amount', 'trip_distance', 'passenger_count'])
    
    # Remove outliers using IQR
    for col in ['total_amount', 'trip_distance', 'trip_duration']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    
    return df

def preprocess_data(df):
    """Create preprocessing pipeline for numerical and categorical features."""
    features = ['trip_distance', 'passenger_count', 'pickup_hour', 
                'is_weekend', 'is_night', 'is_rush_hour', 'trip_duration']
    categorical_features = ['RatecodeID', 'payment_type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor, features, categorical_features