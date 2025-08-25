# forecasting/predictive_analyzer.py
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalyzer:
    def __init__(self):
        self.models = {
            'prophet': Prophet(),
            'xgb': XGBRegressor(),
            'random_forest': RandomForestRegressor()
        }
        self.forecast_results = {}
    
    def prepare_timeseries_data(self, features, target_metric, system):
        """Prepare time series data for forecasting"""
        system_data = features[features['system'] == system]
        ts_data = system_data[['timestamp', target_metric]].copy()
        ts_data = ts_data.set_index('timestamp').resample('1H').mean().ffill()
        
        return ts_data
    
    def train_prophet(self, ts_data, periods=24):
        """Train Facebook Prophet model"""
        prophet_df = ts_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet()
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=periods, freq='H')
        forecast = model.predict(future)
        
        return model, forecast
    
    def train_xgboost(self, ts_data, forecast_horizon=24):
        """Train XGBoost model for time series forecasting"""
        # Create lag features
        for lag in range(1, 25):
            ts_data[f'lag_{lag}'] = ts_data['y'].shift(lag)
        
        ts_data = ts_data.dropna()
        
        # Split features and target
        X = ts_data.drop('y', axis=1)
        y = ts_data['y']
        
        # Train-test split
        split_idx = int(len(ts_data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        return model, predictions, y_test.values
    
    def calculate_risk_score(self, predictions, thresholds):
        """Calculate risk score based on predictions"""
        risk_score = 0
        for metric, prediction in predictions.items():
            if prediction > thresholds[metric]['critical']:
                risk_score += 10
            elif prediction > thresholds[metric]['warning']:
                risk_score += 5
        
        return min(risk_score, 10)  # Normalize to 0-10 scale
    
    def run_forecasts(self, features, systems, metrics):
        """Run forecasting for all systems and metrics"""
        results = {}
        
        for system in systems:
            results[system] = {}
            for metric in metrics:
                try:
                    # Prepare data
                    ts_data = self.prepare_timeseries_data(features, metric, system)
                    
                    if len(ts_data) < 24:  # Need at least 24 hours of data
                        continue
                    
                    # Train models
                    prophet_model, prophet_forecast = self.train_prophet(ts_data)
                    xgb_model, xgb_predictions, y_test = self.train_xgboost(ts_data)
                    
                    # Calculate errors
                    prophet_mae = mean_absolute_error(y_test, prophet_forecast['yhat'].values[-len(y_test):])
                    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
                    
                    # Store results
                    results[system][metric] = {
                        'prophet_forecast': prophet_forecast,
                        'xgb_predictions': xgb_predictions,
                        'prophet_mae': prophet_mae,
                        'xgb_mae': xgb_mae,
                        'last_value': ts_data['y'].iloc[-1]
                    }
                    
                except Exception as e:
                    print(f"Error forecasting {metric} for {system}: {e}")
        
        self.forecast_results = results
        return results