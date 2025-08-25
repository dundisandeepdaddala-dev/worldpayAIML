# feature_store/feature_engine.py
import pandas as pd
import numpy as np
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngine:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.feature_store = pd.DataFrame()
    
    def enrich_logs(self, log_data):
        """Enrich raw logs with additional features"""
        df = pd.DataFrame(log_data)
        
        # Extract features from timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical features
        categorical_features = ['system', 'level', 'namespace', 'region']
        for feature in categorical_features:
            if feature in df.columns:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    df[feature] = self.encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    df[feature] = self.encoders[feature].transform(df[feature].astype(str))
        
        # Text features from log messages
        if 'message' in df.columns:
            tfidf_features = self.vectorizer.fit_transform(df['message']).toarray()
            tfidf_df = pd.DataFrame(tfidf_features, 
                                  columns=[f'message_tfidf_{i}' for i in range(tfidf_features.shape[1])])
            df = pd.concat([df, tfidf_df], axis=1)
        
        # Normalize numerical features
        numerical_features = ['response_time', 'user_id']  # Add more as needed
        for feature in numerical_features:
            if feature in df.columns:
                if feature not in self.scalers:
                    self.scalers[feature] = StandardScaler()
                    df[feature] = self.scalers[feature].fit_transform(df[[feature]])
                else:
                    df[feature] = self.scalers[feature].transform(df[[feature]])
        
        # Store features
        self.feature_store = pd.concat([self.feature_store, df], ignore_index=True)
        
        return df
    
    def get_timeseries_features(self, window_size=10):
        """Create time-series features from feature store"""
        if len(self.feature_store) < window_size:
            return None
        
        # Group by system and create rolling features
        features = []
        for system in self.feature_store['system'].unique():
            system_data = self.feature_store[self.feature_store['system'] == system]
            
            # Create rolling statistics
            rolling_mean = system_data.rolling(window=window_size).mean()
            rolling_std = system_data.rolling(window=window_size).std()
            
            # Add to features
            system_features = pd.concat([
                system_data, 
                rolling_mean.add_prefix('rolling_mean_'),
                rolling_std.add_prefix('rolling_std_')
            ], axis=1)
            
            features.append(system_features.dropna())
        
        return pd.concat(features)