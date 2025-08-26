import time
import json
import pandas as pd
import numpy as np
import random
import re
import requests
import logging
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class SyntheticSystemLogs:
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def _generate_java_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.7, 0.2, 0.1])[0]
        message = ''
        
        if level == 'ERROR':
            message = random.choice([
                'Database connection timeout',
                'OutOfMemoryError: Java heap space',
                'NullPointerException in service layer'
            ])
        elif level == 'WARN':
            message = random.choice([
                'High memory usage detected',
                'Slow database queries',
                'Connection pool at 80% capacity'
            ])
        else:
            message = f'Request processed successfully for user {random.randint(1000, 9999)}'
            
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'java_app',
            'level': level,
            'message': message
        }
    
    def _generate_k8s_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.75, 0.15, 0.1])[0]
        message = ''
        
        if level == 'ERROR':
            message = random.choice([
                'Pod crashed due to OOM',
                'Image pull failed',
                'Node not ready'
            ])
        elif level == 'WARN':
            message = random.choice([
                'High memory pressure on node',
                'CPU throttling detected',
                'Network latency increased'
            ])
        else:
            message = 'Pod started successfully'
            
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'kubernetes',
            'level': level,
            'message': message
        }
    
    def _generate_cobol_logs(self):
        level = random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.8, 0.15, 0.05])[0]
        message = ''
        
        if level == 'ERROR':
            message = random.choice([
                'ABEND: U4038 System error',
                'Dataset mount failed',
                'CICS transaction abended'
            ])
        elif level == 'WARN':
            message = random.choice([
                'Batch job running longer than expected',
                'Storage allocation nearing limit',
                'Tape drive response slow'
            ])
        else:
            message = 'Batch job completed successfully'
            
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'cobol_mainframe',
            'level': level,
            'message': message
        }
    
    def collect_data(self):
        """Collect logs from all systems"""
        logs = []
        logs.append(self._generate_java_logs())
        logs.append(self._generate_k8s_logs())
        logs.append(self._generate_cobol_logs())
        
        # 20% chance of metric anomaly
        metric_anomaly = random.random() < 0.2
        
        return logs, metric_anomaly

class AIObservabilityPlatform:
    def __init__(self):
        self.synthetic_generator = SyntheticSystemLogs(verbose=True)
        self.systems = ['java_app', 'kubernetes', 'cobol_mainframe']
        self.anomaly_history = []
        self.feedback_queue = []
        self.is_monitoring = False
        self.monitoring_thread = None
    
    def collect_and_process_data(self):
        """Collect and process data from all systems"""
        print("Collecting and processing data...")
        
        # Collect data from synthetic generator
        logs, metric_anomaly = self.synthetic_generator.collect_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(logs)
        
        # Add basic features
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['is_error'] = (df['level'] == 'ERROR').astype(int)
        df['is_warning'] = (df['level'] == 'WARN').astype(int)
        
        return df, metric_anomaly
    
    def run_predictive_analysis(self, features):
        """Run predictive analysis on features"""
        print("Running predictive analysis...")
        
        if features.empty:
            return {}, {}
        
        # Simple risk score based on recent error rate
        risk_scores = {}
        
        for system in self.systems:
            system_data = features[features['system'] == system]
            if len(system_data) > 0:
                error_count = len(system_data[system_data['level'] == 'ERROR'])
                risk_scores[system] = min(error_count * 2, 10)  # Scale to 0-10
        
        return {}, risk_scores
    
    def detect_anomalies(self, features, metric_anomaly):
        """Detect anomalies using simple threshold-based approach"""
        print("Detecting anomalies...")
        
        anomalies = {}
        
        # Check for metric anomalies
        if metric_anomaly:
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
                'DatabaseConnectionTimeout': r'Database connection timeout'
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
            'CPUThrottling': 7.0,
            'NodeNotReady': 8.0,
            'StorageWarning': 6.5
        }
        
        score = base_scores.get(issue_type, 7.0)
        
        # Adjust based on log level
        if log_level == 'WARN':
            score *= 0.7
        
        return min(score, 10.0)

    def _handle_anomaly(self, system, anomaly_info):
        """Handle detected anomaly with specific guidance"""
        print(f"🔍 Handling anomaly for {system}: {anomaly_info['type']}")
        
        # Generate simple guidance
        guidance = self._generate_guidance(anomaly_info)
        
        # Store anomaly info
        anomaly_record = {
            'info': anomaly_info,
            'rca_result': {'comprehensive_guidance': guidance},
            'timestamp': datetime.now(),
            'resolved': False,
            'feedback_provided': False
        }
        self.anomaly_history.append(anomaly_record)
        
        print(f"\n🚨 {anomaly_info['type'].replace('_', ' ').upper()} DETECTED in {system.upper()}")
        print(f"   Message: {anomaly_info['message']}")
        print(f"   Time: {anomaly_info['timestamp']}")
        print(f"   Severity Score: {anomaly_info['score']:.2f}/10")
        
        print("\n🔍 GUIDANCE:")
        print(guidance)
        print("-" * 60)
        
        # Add to feedback queue
        self.feedback_queue.append(anomaly_record)
    
    def _generate_guidance(self, anomaly_info):
        """Generate simple guidance for anomalies"""
        system = anomaly_info.get('system', 'unknown')
        issue_type = anomaly_info.get('type', 'unknown')
        
        guidance_templates = {
            'java_app': {
                'NullPointerException': """IMMEDIATE ACTIONS:
1. Check stack trace to identify exact line causing NPE
2. Review recent code changes around that location
3. Add null checks: if (object != null) { object.method(); }""",
                'OutOfMemoryError': """IMMEDIATE ACTIONS:
1. Increase JVM heap size: -Xmx4g
2. Restart the application
3. Check for memory leaks"""
            },
            'kubernetes': {
                'CPUThrottling': """IMMEDIATE ACTIONS:
1. Increase CPU limits for the deployment
2. Add more replicas to distribute load
3. Check node CPU capacity"""
            },
            'cobol_mainframe': {
                'StorageWarning': """IMMEDIATE ACTIONS:
1. Clean up temporary datasets
2. Compress existing datasets
3. Request temporary storage increase"""
            }
        }
        
        system_guidance = guidance_templates.get(system, {})
        return system_guidance.get(issue_type, "Investigate the issue and check system logs")
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        print("Starting AI Observability Platform...")
        print("Systems monitored:", self.systems)
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        self.is_monitoring = True
        
        while self.is_monitoring:
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
                time.sleep(10)
                
            except KeyboardInterrupt:
                print("\nStopping AI Observability Platform...")
                self.is_monitoring = False
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(10)

# Initialize the platform
platform = AIObservabilityPlatform()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is reachable (but don't fail if it's not)
        try:
            ollama_response = requests.get("http://ollama:11434/api/tags", timeout=5)
            ollama_healthy = ollama_response.status_code == 200
        except:
            ollama_healthy = False
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "services": {
                    "ollama": ollama_healthy,
                    "platform": True,
                    "monitoring_active": platform.is_monitoring
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for Prometheus"""
    metrics_data = {
        "anomalies_total": len(platform.anomaly_history),
        "feedback_queue_size": len(platform.feedback_queue),
        "monitoring_active": 1 if platform.is_monitoring else 0,
        "systems_monitored": len(platform.systems)
    }
    
    return JSONResponse(status_code=200, content=metrics_data)

@app.get("/status")
async def platform_status():
    """Get current platform status"""
    return JSONResponse(
        status_code=200,
        content={
            "systems": platform.systems,
            "anomaly_count": len(platform.anomaly_history),
            "feedback_queue_size": len(platform.feedback_queue),
            "monitoring_active": platform.is_monitoring
        }
    )

@app.get("/anomalies")
async def get_anomalies(limit: int = 10):
    """Get recent anomalies"""
    recent_anomalies = []
    for anomaly in platform.anomaly_history[-limit:]:
        recent_anomalies.append({
            "system": anomaly['info'].get('system', 'unknown'),
            "type": anomaly['info'].get('type', 'unknown'),
            "message": anomaly['info'].get('message', ''),
            "timestamp": anomaly['info'].get('timestamp', ''),
            "score": anomaly['info'].get('score', 0),
            "resolved": anomaly.get('resolved', False)
        })
    
    return JSONResponse(status_code=200, content={"anomalies": recent_anomalies})

@app.post("/monitoring/start")
async def start_monitoring():
    """Start the monitoring process"""
    if platform.is_monitoring:
        return JSONResponse(
            status_code=400,
            content={"message": "Monitoring is already running"}
        )
    
    # Start monitoring in a separate thread
    def run_monitoring():
        platform.is_monitoring = True
        try:
            platform.run_continuous_monitoring()
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
        finally:
            platform.is_monitoring = False
    
    platform.monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
    platform.monitoring_thread.start()
    
    return JSONResponse(
        status_code=200,
        content={"message": "Monitoring started successfully"}
    )

@app.post("/monitoring/stop")
async def stop_monitoring():
    """Stop the monitoring process"""
    platform.is_monitoring = False
    return JSONResponse(
        status_code=200,
        content={"message": "Monitoring stop signal sent"}
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Observability Platform API",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "status": "/status",
            "anomalies": "/anomalies?limit=10",
            "start_monitoring": "/monitoring/start (POST)",
            "stop_monitoring": "/monitoring/stop (POST)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)