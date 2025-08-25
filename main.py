# Updated main.py
import time
import json
import pandas as pd
from datetime import datetime
from synthetic_data.multi_system_logs import SyntheticSystemLogs
from feature_store.feature_engine import FeatureEngine
from forecasting.predictive_analyzer import PredictiveAnalyzer
from anomaly_detection.deep_anomaly import DeepAnomalyDetector
from rca.llm_analyzer import LLMAnalyzer
from feedback_loop.model_retrainer import ModelRetrainer
from visualization.dashboard import ObservabilityDashboard
import sys
import os

class AIObservabilityPlatform:
    def __init__(self):
        self.synthetic_generator = SyntheticSystemLogs()
        self.feature_engine = FeatureEngine()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.anomaly_detector = DeepAnomalyDetector()
        self.llm_analyzer = LLMAnalyzer()
        self.model_retrainer = ModelRetrainer()
        self.dashboard = ObservabilityDashboard()
        
        self.systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        self.metrics = ['cpu_usage', 'memory_usage', 'error_count']
        
        self.anomaly_history = []
        self.retrain_interval = 24 * 3600  # Retrain every 24 hours
        self.last_retrain_time = time.time()
    
    # [Keep all your existing methods]
    
    def run_feedback_loop(self):
        """Run feedback loop for model retraining"""
        current_time = time.time()
        
        # Check if it's time to retrain
        if current_time - self.last_retrain_time >= self.retrain_interval:
            print("Running feedback loop and model retraining...")
            
            # Prepare data for retraining
            for system in self.systems:
                system_data = self.feature_engine.feature_store[
                    self.feature_engine.feature_store['system'] == system
                ]
                
                if len(system_data) > 0:
                    numerical_data = system_data.select_dtypes(include=[np.number])
                    
                    # For anomaly detection retraining
                    self.anomaly_detector.train_models({system: numerical_data})
                    
                    # For predictive model retraining
                    ts_features = self.feature_engine.get_timeseries_features()
                    if ts_features is not None:
                        self.predictive_analyzer.run_forecasts(
                            ts_features, [system], self.metrics
                        )
            
            self.last_retrain_time = current_time
            print("Model retraining completed")
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        print("Starting AI Observability Platform...")
        print("Systems monitored:", self.systems)
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n=== Iteration {iteration} - {datetime.now()} ===")
            
            try:
                # Step 1: Collect and process data
                features = self.collect_and_process_data()
                
                # Step 2: Predictive analysis
                forecasts, risk_scores = self.run_predictive_analysis(features)
                
                # Step 3: Anomaly detection
                anomalies = self.detect_anomalies(features)
                
                # Step 4: Run feedback loop if needed
                self.run_feedback_loop()
                
                # Step 5: Update dashboard
                self.dashboard.run(risk_scores, anomalies, forecasts, self.anomaly_history)
                
                # Wait before next iteration
                time.sleep(300)  # 5 minutes between iterations
                
            except KeyboardInterrupt:
                print("\nStopping AI Observability Platform...")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    platform = AIObservabilityPlatform()
    platform.run_continuous_monitoring()