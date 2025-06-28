import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
from utils import engineer_features, clean_data, preprocess_data

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load the taxi trip dataset."""
    try:
        df = pd.read_csv('taxi_trip_data.csv')
        print("Data loaded successfully.")
        print("\nInspecting datetime columns:")
        print(df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].head())
        print("\nData types:")
        print(df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].dtypes)
        return df
    except FileNotFoundError:
        print("Error: Dataset file not found. Please provide the correct path.")
        return None

def perform_eda(df):
    """Perform exploratory data analysis and save visualizations."""
    plt.figure(figsize=(15, 10))
    
    # Fare vs Distance
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='trip_distance', y='total_amount')
    plt.title('Fare vs Trip Distance')
    plt.savefig('fare_vs_distance.png')
    
    # Fare distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['total_amount'], bins=50)
    plt.title('Distribution of Total Fare')
    plt.savefig('fare_distribution.png')
    
    # Fare by hour
    plt.subplot(2, 2, 3)
    sns.boxplot(x='pickup_hour', y='total_amount', data=df)
    plt.title('Fare by Pickup Hour')
    plt.savefig('fare_by_hour.png')
    
    # Fare by day
    plt.subplot(2, 2, 4)
    sns.boxplot(x='pickup_day', y='total_amount', data=df)
    plt.title('Fare by Pickup Day')
    plt.savefig('fare_by_day.png')
    
    plt.tight_layout()
    plt.close()

def train_and_evaluate_models(X, y, preprocessor):
    """Train and evaluate multiple regression models."""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    best_model = None
    best_r2 = -float('inf')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        # Save best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(pipeline, f)
    
    return pd.DataFrame(results)

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Feature engineering
    df = engineer_features(df)
    
    # Data cleaning
    df = clean_data(df)
    
    # Perform EDA
    perform_eda(df)
    
    # prepara features and target
    X = df.drop('total_amount', axis=1)
    y = df['total_amount']
    
    # Preprocess data
    preprocessor, _, _ = preprocess_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(X, y, preprocessor)
    print("\nModel Performance Metrics:")
    print(results)
    
    # Save results
    results.to_csv('model_performance.csv', index=False)

if __name__ == '__main__':
    main()