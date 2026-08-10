import os
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_URL = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"
LOCAL_PATH = "data/sales.csv"

def download_data(url=DATA_URL, path=LOCAL_PATH):
    """Downloads the Superstore sales dataset if it doesn't already exist locally."""
    if os.path.exists(path):
        print(f"Dataset already exists at {path}")
        return path
    
    print(f"Downloading dataset from {url}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Dataset successfully downloaded and saved to {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Return fallback or raise
        raise e
    return path

def load_and_clean_data(path=LOCAL_PATH, uploaded_file=None):
    """
    Loads raw sales data (from local file or uploaded file),
    applies column mapping, parses dates, handles missing values/duplicates,
    and removes extreme outliers.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        if not os.path.exists(path):
            download_data(path=path)
        df = pd.read_csv(path)
        
    print(f"Raw dataset shape: {df.shape}")
    
    # Standardize column names dynamically
    df = map_columns(df)
    
    # Required columns check
    required_cols = ['Date', 'Product', 'Category', 'Quantity', 'Sales']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Clean and parse Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Standardize types
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df = df.dropna(subset=['Quantity', 'Sales'])
    
    # Drop duplicate records
    df = df.drop_duplicates()
    
    # Filter out non-positive quantities and sales
    df = df[(df['Quantity'] > 0) & (df['Sales'] > 0)]
    
    # Recalculate Price if not present
    if 'Price' not in df.columns:
        df['Price'] = df['Sales'] / df['Quantity']
    else:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(df['Sales'] / df['Quantity'])
        
    # Region fallback
    if 'Store/Region' not in df.columns:
        df['Store/Region'] = 'Unknown'
        
    # Handle Outliers (clip at 99.9th percentile)
    q_sales = df['Sales'].quantile(0.999)
    q_qty = df['Quantity'].quantile(0.999)
    df = df[(df['Sales'] <= q_sales) & (df['Quantity'] <= q_qty)]
    
    # Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Cleaned dataset shape: {df.shape}")
    return df

def map_columns(df):
    """Maps custom or messy column names to standardized columns."""
    mappings = {
        'Date': ['date', 'order date', 'order_date', 'transaction date', 'transaction_date', 'time'],
        'Product': ['product', 'product name', 'product_name', 'item', 'item name', 'item_name', 'product id', 'product_id'],
        'Category': ['category', 'product category', 'product_category', 'dept', 'department'],
        'Quantity': ['quantity', 'qty', 'units', 'count', 'number of items', 'volume'],
        'Sales': ['sales', 'revenue', 'amount', 'sales amount', 'sales_amount', 'total sales', 'total_sales', 'price_total'],
        'Store/Region': ['region', 'store', 'store/region', 'branch', 'location', 'state', 'city']
    }
    
    renamed = {}
    for target, options in mappings.items():
        for col in df.columns:
            if col.lower() == target.lower() or col.lower() in options:
                renamed[col] = target
                break
    
    # If no exact match, try partial match for important ones
    for target, options in mappings.items():
        if target not in renamed.values():
            for col in df.columns:
                if col not in renamed:
                    if any(opt in col.lower() for opt in options):
                        renamed[col] = target
                        break
                        
    return df.rename(columns=renamed)

def aggregate_daily_sales(df):
    """Aggregates transactional sales data to the daily level, filling date gaps."""
    # Daily aggregation
    df_daily = df.groupby('Date')[['Sales', 'Quantity']].sum().reset_index()
    df_daily.set_index('Date', inplace=True)
    
    # Reindex to fill missing dates with 0 sales
    all_days = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
    df_daily = df_daily.reindex(all_days, fill_value=0)
    df_daily.index.name = 'Date'
    df_daily.reset_index(inplace=True)
    
    return df_daily

def engineer_features(df_daily):
    """Generates time-series and calendar features on daily aggregated data."""
    df_features = df_daily.copy()
    
    # Calendar features
    df_features['Year'] = df_features['Date'].dt.year
    df_features['Month'] = df_features['Date'].dt.month
    df_features['Day'] = df_features['Date'].dt.day
    df_features['Week'] = df_features['Date'].dt.isocalendar().week.astype(int)
    df_features['Quarter'] = df_features['Date'].dt.quarter
    df_features['Dayofweek'] = df_features['Date'].dt.dayofweek
    
    # Lag features (prevent lookahead bias by shifting)
    df_features['lag_1'] = df_features['Sales'].shift(1)
    df_features['lag_7'] = df_features['Sales'].shift(7)
    
    # Rolling features (shift by 1 first so they use historical data only)
    df_features['rolling_mean_7'] = df_features['Sales'].shift(1).rolling(window=7).mean()
    df_features['rolling_mean_30'] = df_features['Sales'].shift(1).rolling(window=30).mean()
    
    # Drop rows with NaN values resulting from lags/rolling
    df_features = df_features.dropna().reset_index(drop=True)
    return df_features

def train_and_evaluate_models(df_features):
    """
    Splits data chronologically, trains LR, RF, and XGBoost models,
    and returns models, metrics, and test-set predictions.
    """
    # 80-20 Time-series split
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    feature_cols = [
        'Year', 'Month', 'Day', 'Week', 'Quarter', 'Dayofweek',
        'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_mean_30'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Sales']
    X_test = test_df[feature_cols]
    y_test = test_df['Sales']
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    # Train and evaluate
    evaluation_results = []
    predictions = {'Date': test_df['Date'].values, 'Actual': y_test.values}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        # Avoid negative predictions for sales
        preds = np.clip(preds, 0, None)
        predictions[name] = preds
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # MAPE (avoid division by zero)
        mask = y_test > 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
        else:
            mape = 0.0
            
        evaluation_results.append({
            'Model': name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2)
        })
        
    metrics_df = pd.DataFrame(evaluation_results)
    predictions_df = pd.DataFrame(predictions)
    
    return models, metrics_df, predictions_df

def forecast_future(models, historical_daily_df, horizon=30, selected_model_name='XGBoost'):
    """
    Performs recursive forecasting for a specified future horizon
    by iteratively generating lag and rolling features.
    """
    model = models[selected_model_name]
    
    # Needs at least 30 days of history for rolling features
    last_history = historical_daily_df.tail(30).copy()
    
    # List to store prediction records
    forecast_records = []
    
    # Start forecasting from the next day
    current_date = historical_daily_df['Date'].max()
    
    for i in range(horizon):
        current_date += pd.Timedelta(days=1)
        
        # Create calendar features
        year = current_date.year
        month = current_date.month
        day = current_date.day
        week = current_date.isocalendar()[1]
        quarter = (month - 1) // 3 + 1
        dayofweek = current_date.dayofweek
        
        # Lag features from history
        lag_1 = last_history.iloc[-1]['Sales']
        lag_7 = last_history.iloc[-7]['Sales']
        
        # Rolling features
        rolling_7 = last_history.iloc[-7:]['Sales'].mean()
        rolling_30 = last_history.iloc[-30:]['Sales'].mean()
        
        # Construct feature vector
        features = pd.DataFrame([{
            'Year': year,
            'Month': month,
            'Day': day,
            'Week': week,
            'Quarter': quarter,
            'Dayofweek': dayofweek,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'rolling_mean_7': rolling_7,
            'rolling_mean_30': rolling_30
        }])
        
        # Run prediction
        pred_sales = model.predict(features)[0]
        pred_sales = max(0.0, float(pred_sales)) # clip to non-negative
        
        # Record forecasted value
        forecast_records.append({
            'Date': current_date,
            'Predicted Sales': round(pred_sales, 2)
        })
        
        # Append prediction to history to use in the next iterations
        new_row = pd.DataFrame([{'Date': current_date, 'Sales': pred_sales, 'Quantity': 0.0}])
        last_history = pd.concat([last_history, new_row], ignore_index=True).tail(30)
        
    return pd.DataFrame(forecast_records)

if __name__ == "__main__":
    # Test script execution
    print("Testing data downloading and training pipeline...")
    path = download_data()
    df = load_and_clean_data(path)
    df_daily = aggregate_daily_sales(df)
    df_features = engineer_features(df_daily)
    models, metrics, predictions = train_and_evaluate_models(df_features)
    print("\nModel Evaluation Results on Test Set:")
    print(metrics)
    
    print("\nGenerating 7-day forecast using XGBoost...")
    forecast = forecast_future(models, df_daily, horizon=7, selected_model_name='XGBoost')
    print(forecast)
    print("\nPipeline test completed successfully!")
