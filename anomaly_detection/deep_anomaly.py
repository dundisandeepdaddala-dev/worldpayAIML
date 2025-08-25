# anomaly_detection/deep_anomaly.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DeepAnomalyDetector:
    def __init__(self, sequence_length=10, feature_dim=20):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scalers = {}
        self.models = {}
        self.reconstruction_errors = {}
    
    def create_autoencoder(self):
        """Create LSTM autoencoder model"""
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        
        # Encoder
        encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
        encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
        decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.feature_dim))(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM input"""
        sequences = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i+sequence_length])
        return np.array(sequences)
    
    def train_models(self, features_by_system):
        """Train autoencoder for each system"""
        for system, features in features_by_system.items():
            # Scale features
            if system not in self.scalers:
                self.scalers[system] = StandardScaler()
                scaled_features = self.scalers[system].fit_transform(features)
            else:
                scaled_features = self.scalers[system].transform(features)
            
            # Create sequences
            sequences = self.create_sequences(scaled_features, self.sequence_length)
            
            if len(sequences) < 10:  # Need enough sequences
                continue
            
            # Create and train autoencoder
            autoencoder = self.create_autoencoder()
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            autoencoder.fit(
                sequences, sequences,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.models[system] = autoencoder
            
            # Calculate reconstruction error
            reconstructions = autoencoder.predict(sequences, verbose=0)
            mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
            self.reconstruction_errors[system] = mse
    
    def detect_anomalies(self, new_features, system, threshold_std=2):
        """Detect anomalies in new data"""
        if system not in self.models:
            return None
        
        # Scale features
        scaled_features = self.scalers[system].transform(new_features)
        
        # Create sequences
        sequences = self.create_sequences(scaled_features, self.sequence_length)
        
        if len(sequences) == 0:
            return None
        
        # Get reconstructions and errors
        reconstructions = self.models[system].predict(sequences, verbose=0)
        mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
        
        # Calculate threshold
        train_errors = self.reconstruction_errors[system]
        threshold = np.mean(train_errors) + threshold_std * np.std(train_errors)
        
        # Detect anomalies
        anomalies = mse > threshold
        anomaly_scores = mse / threshold  # Normalized anomaly score
        
        return {
            'anomalies': anomalies,
            'scores': anomaly_scores,
            'reconstruction_errors': mse,
            'threshold': threshold
        }