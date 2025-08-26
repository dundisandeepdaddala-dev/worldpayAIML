# main.py
import time
import json
import pandas as pd
import numpy as np
import random
import re
from datetime import datetime
from synthetic_data.multi_system_logs import SyntheticSystemLogs
from feature_store.feature_engine import FeatureEngine
from true_guidance.guided_rca import GuidedRCA

class AIObservabilityPlatform:
    def __init__(self):
        self.synthetic_generator = SyntheticSystemLogs(verbose=True)
        self.feature_engine = FeatureEngine()
        
        # Use guided RCA analyzer instead of simple LLM analyzer
        self.guided_rca = GuidedRCA()
        
        self.systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        self.metrics = ['cpu_usage', 'memory_usage', 'error_count']
        
        self.anomaly_history = []
    
    def collect_and_process_data(self):
        """Collect and process data from all systems"""
        print("Collecting and processing data...")
        
        # Collect data from synthetic generator
        logs, metric_anomaly = self.synthetic_generator.collect_data()
        
        # Add logs to knowledge base for RCA (replaces the vector DB)
        for log in logs:
            # We'll use the guided RCA's knowledge graph instead of a simple vector DB
            pass
        
        # Enrich logs with features
        enriched_features = self.feature_engine.enrich_logs(logs)
        
        return enriched_features, metric_anomaly
    
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
    
    def detect_anomalies(self, features, metric_anomaly):
        """Detect anomalies using simple threshold-based approach"""
        print("Detecting anomalies...")
        
        anomalies = {}
        
        # Check for metric anomalies
        if metric_anomaly:
            # Determine which system has the metric anomaly
            metric_context = self._get_metric_context(features)
            if metric_context:
                anomalies['metrics'] = {
                    'type': f'Metric_Anomaly_{metric_context}',
                    'message': f'Abnormal metric values detected in {metric_context}',
                    'timestamp': datetime.now().isoformat(),
                    'score': 8.0,
                    'system': metric_context
                }
            else:
                anomalies['metrics'] = {
                    'type': 'Metric_Anomaly_General',
                    'message': 'Abnormal metric values detected across systems',
                    'timestamp': datetime.now().isoformat(),
                    'score': 7.0,
                    'system': 'general'
                }
        
        # Check for specific log patterns
        for system in self.systems:
            system_data = features[features['system'] == system]
            if len(system_data) == 0:
                continue
                
            # Check all logs in this batch
            for _, log in system_data.iterrows():
                if log['level'] in ['ERROR', 'WARN']:
                    detected_issue = self._classify_issue(log['message'], system)
                    if detected_issue and system not in anomalies:
                        anomalies[system] = {
                            'type': detected_issue,
                            'message': log['message'],
                            'timestamp': log['timestamp'],
                            'score': self._calculate_severity_score(detected_issue, log['level']),
                            'system': system
                        }
        
        return anomalies

    def _classify_issue(self, message, system):
        """Classify the specific issue from the log message"""
        known_patterns = {
            'java_app': {
                'NullPointerException': r'NullPointerException',
                'OutOfMemoryError': r'OutOfMemoryError',
                'DatabaseConnectionTimeout': r'Database connection timeout',
                'ConnectionPoolWarning': r'Connection pool at.*capacity'
            },
            'kubernetes': {
                'CPUThrottling': r'CPU throttling detected',
                'NodeNotReady': r'Node not ready'
            },
            'cobol_mainframe': {
                'StorageWarning': r'Storage allocation nearing limit'
            }
        }
        
        if system not in known_patterns:
            return None
            
        for issue_type, pattern in known_patterns[system].items():
            if re.search(pattern, message, re.IGNORECASE):
                return issue_type
                
        return None

    def _calculate_severity_score(self, issue_type, log_level):
        """Calculate severity score based on issue type and log level"""
        base_scores = {
            'NullPointerException': 8.0,
            'OutOfMemoryError': 9.0,
            'DatabaseConnectionTimeout': 7.5,
            'ConnectionPoolWarning': 5.5,
            'CPUThrottling': 7.0,
            'NodeNotReady': 8.0,
            'StorageWarning': 6.5
        }
        
        # Default score if issue type not in base_scores
        score = base_scores.get(issue_type, 7.0)
        
        # Adjust based on log level
        if log_level == 'WARN':
            score *= 0.7  # Reduce score for warnings
        
        return min(score, 10.0)  # Cap at 10.0

    def _get_metric_context(self, features):
        """Determine which system is most likely causing metric anomalies"""
        # Look for recent errors to provide context
        recent_errors = {}
        
        for system in self.systems:
            system_data = features[features['system'] == system]
            error_count = len(system_data[system_data['level'] == 'ERROR'])
            recent_errors[system] = error_count
        
        # Return the system with the most recent errors
        if recent_errors:
            return max(recent_errors.items(), key=lambda x: x[1])[0]
        
        return None

    def _handle_anomaly(self, system, anomaly_info):
        """Handle detected anomaly with specific guidance"""
        # Perform guided RCA with specific context
        print(f"🔍 Performing guided RCA for {system} {anomaly_info['type']}...")
        
        # Prepare incident data with specific context
        incident_data = {
            'system': system,
            'type': anomaly_info['type'],
            'message': anomaly_info['message'],
            'timestamp': anomaly_info['timestamp'],
            'severity': 'high' if anomaly_info['score'] > 7 else 'medium',
            'metrics': self._get_current_metrics(system)
        }
        
        # Get guided RCA
        rca_result = self.guided_rca.analyze_incident(incident_data)
        
        # Store anomaly info
        self.anomaly_history.append({
            'info': anomaly_info,
            'rca_result': rca_result,
            'timestamp': datetime.now()
        })
        
        print(f"\n🚨 {anomaly_info['type'].replace('_', ' ').upper()} DETECTED in {system.upper()}")
        print(f"   Message: {anomaly_info['message']}")
        print(f"   Time: {anomaly_info['timestamp']}")
        print(f"   Severity Score: {anomaly_info['score']:.2f}/10")
        
        # Display the comprehensive guidance
        print("\n🔍 ROOT CAUSE ANALYSIS & GUIDANCE:")
        print(rca_result['comprehensive_guidance'])
        print("-" * 60)
        
        # Trigger alert
        self._trigger_alert(anomaly_info, rca_result['comprehensive_guidance'])
    
    def _get_current_metrics(self, system):
        """Get current metrics for the system"""
        # This would query your metrics database in a real implementation
        # For now, return synthetic metrics
        return {
            'cpu_usage': random.uniform(20, 90),
            'memory_usage': random.uniform(30, 85),
            'error_rate': random.uniform(0.1, 5.0)
        }
    
    def _trigger_alert(self, anomaly_info, rca_result):
        """Trigger alert based on anomaly"""
        print(f"⏰ ALERT: {anomaly_info['type'].upper()} in {anomaly_info.get('system', 'unknown')}")
        print(f"   Score: {anomaly_info['score']:.2f}")
        print(f"   Message: {anomaly_info['message']}")
        print(f"   RCA: {rca_result[:100]}...")  # First 100 chars
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        print("Starting AI Observability Platform with Guided RCA...")
        print("Systems monitored:", self.systems)
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        while True:
            iteration += 1
            print(f"\n=== Iteration {iteration} - {datetime.now()} ===")
            
            try:
                # Step 1: Collect and process data
                features, metric_anomaly = self.collect_and_process_data()
                
                if features.empty:
                    print("No features collected, skipping iteration")
                    time.sleep(5)
                    continue
                
                # Step 2: Predictive analysis
                forecasts, risk_scores = self.run_predictive_analysis(features)
                
                # Step 3: Anomaly detection
                anomalies = self.detect_anomalies(features, metric_anomaly)
                
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