# main.py
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from synthetic_data.multi_system_logs import SyntheticSystemLogs
from feature_store.feature_engine import FeatureEngine
from forecasting.predictive_analyzer import PredictiveAnalyzer
from anomaly_detection.deep_anomaly import DeepAnomalyDetector
from rca.llm_analyzer import LLMAnalyzer

class AIObservabilityPlatform:
    def __init__(self):
        self.synthetic_generator = SyntheticSystemLogs()
        self.feature_engine = FeatureEngine()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.anomaly_detector = DeepAnomalyDetector()
        self.llm_analyzer = LLMAnalyzer()
        
        self.systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        self.metrics = ['cpu_usage', 'memory_usage', 'error_count']
        
        self.anomaly_history = []
    
    def collect_and_process_data(self):
        """Collect and process data from all systems"""
        print("Collecting and processing data...")
        
        # Collect data from synthetic generator
        logs = self.synthetic_generator.collect_data()
        
        # Add logs to vector DB for RAG
        for log in logs:
            self.llm_analyzer.add_to_vector_db(log)
        
        # Enrich logs with features
        enriched_features = self.feature_engine.enrich_logs(logs)
        
        return enriched_features
    
    def run_predictive_analysis(self, features):
        """Run predictive analysis on features"""
        print("Running predictive analysis...")
        
        # Get time-series features
        ts_features = self.feature_engine.get_timeseries_features()
        
        if ts_features is not None and len(ts_features) > 0:
            # For simplicity, we'll just use a single metric for forecasting
            forecasts = {}
            risk_scores = {}
            
            for system in self.systems:
                system_data = ts_features[ts_features['system'] == system]
                if len(system_data) > 0:
                    # Simple risk score based on recent error rate
                    error_count = len(system_data[system_data['level'] == 'ERROR'])
                    risk_scores[system] = min(error_count * 2, 10)  # Scale to 0-10
                    
                    # Store simple forecast (just the latest values)
                    forecasts[system] = {
                        'last_cpu': system_data['cpu_usage'].iloc[-1] if 'cpu_usage' in system_data.columns else 0,
                        'last_memory': system_data['memory_usage'].iloc[-1] if 'memory_usage' in system_data.columns else 0,
                    }
            
            return forecasts, risk_scores
        
        return {}, {}
    
    def detect_anomalies(self, features):
        """Detect anomalies using simple threshold-based approach"""
        print("Detecting anomalies...")
        
        anomalies = {}
        
        for system in self.systems:
            system_data = features[features['system'] == system]
            if len(system_data) == 0:
                continue
                
            # Simple threshold-based anomaly detection
            latest_log = system_data.iloc[-1]
            
            # Check for error logs
            if latest_log['level'] == 'ERROR':
                anomalies[system] = {
                    'type': 'error',
                    'message': latest_log['message'],
                    'timestamp': latest_log['timestamp'],
                    'score': 8.0  # High score for errors
                }
            
            # Check for warning logs
            elif latest_log['level'] == 'WARN':
                anomalies[system] = {
                    'type': 'warning',
                    'message': latest_log['message'],
                    'timestamp': latest_log['timestamp'],
                    'score': 5.0  # Medium score for warnings
                }
        
        return anomalies
    
    def _handle_anomaly(self, system, anomaly_info):
        """Handle detected anomaly"""
        # Search for similar historical incidents
        similar_logs = self.llm_analyzer.search_similar_logs(anomaly_info['message'])
        
        # Perform RCA
        rca_result = self.llm_analyzer.perform_rca(anomaly_info, similar_logs, {})
        
        # Store anomaly info
        self.anomaly_history.append({
            'info': anomaly_info,
            'rca_result': rca_result,
            'timestamp': datetime.now()
        })
        
        print(f"🚨 Anomaly detected in {system}! Score: {anomaly_info['score']:.2f}")
        print(f"Root cause analysis:\n{rca_result}")
        
        # Trigger alert
        self._trigger_alert(anomaly_info, rca_result)
    
    def _trigger_alert(self, anomaly_info, rca_result):
        """Trigger alert based on anomaly"""
        print(f"⏰ ALERT: {anomaly_info['type'].upper()} in {anomaly_info.get('system', 'unknown')}")
        print(f"   Score: {anomaly_info['score']:.2f}")
        print(f"   Message: {anomaly_info['message']}")
        print(f"   RCA: {rca_result[:100]}...")  # First 100 chars
    
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
                
                if features.empty:
                    print("No features collected, skipping iteration")
                    time.sleep(5)
                    continue
                
                # Step 2: Predictive analysis
                forecasts, risk_scores = self.run_predictive_analysis(features)
                
                # Step 3: Anomaly detection
                anomalies = self.detect_anomalies(features)
                
                # Step 4: Handle anomalies
                for system, anomaly_info in anomalies.items():
                    anomaly_info['system'] = system
                    self._handle_anomaly(system, anomaly_info)
                
                # Step 5: Display results
                print("\n📊 Current Risk Scores:")
                for system, score in risk_scores.items():
                    print(f"   {system}: {score}/10")
                
                if anomalies:
                    print(f"\n🚨 Anomalies detected: {len(anomalies)}")
                else:
                    print("\n✅ No anomalies detected")
                
                # Wait before next iteration
                time.sleep(10)  # 10 seconds between iterations for demo
                
            except KeyboardInterrupt:
                print("\nStopping AI Observability Platform...")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)  # Wait before retrying

if __name__ == "__main__":
    platform = AIObservabilityPlatform()
    platform.run_continuous_monitoring()