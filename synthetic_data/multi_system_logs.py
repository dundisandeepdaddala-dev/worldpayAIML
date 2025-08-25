# synthetic_data/multi_system_logs.py
import random
import time
import json
from datetime import datetime
from faker import Faker
import numpy as np
from prometheus_client import start_http_server, Gauge, Counter, Histogram

fake = Faker()

class SyntheticSystemLogs:
    def __init__(self):
        self.systems = {
            'java_app': self._generate_java_logs,
            'kubernetes': self._generate_k8s_logs,
            'cobol_mainframe': self._generate_cobol_logs
        }
        
        # Initialize metrics
        self._setup_metrics()
        
    def _setup_metrics(self):
        # System metrics
        self.metrics = {
            'java_cpu': Gauge('java_app_cpu_usage', 'CPU usage for Java application'),
            'java_memory': Gauge('java_app_memory_usage', 'Memory usage for Java application'),
            'java_errors': Counter('java_app_error_count', 'Error count for Java application'),
            'k8s_cpu': Gauge('k8s_cpu_usage', 'CPU usage for Kubernetes cluster'),
            'k8s_memory': Gauge('k8s_memory_usage', 'Memory usage for Kubernetes cluster'),
            'k8s_pods': Gauge('k8s_pod_count', 'Pod count in Kubernetes cluster'),
            'cobol_cpu': Gauge('cobol_cpu_usage', 'CPU usage for COBOL mainframe'),
            'cobol_jobs': Counter('cobol_jobs_processed', 'Jobs processed by COBOL mainframe'),
            'cobol_errors': Counter('cobol_error_count', 'Error count for COBOL mainframe')
        }
    
    def _generate_java_logs(self):
        log = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'java_app',
            'level': random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.7, 0.2, 0.1])[0],
            'message': '',
            'trace_id': fake.uuid4(),
            'user_id': fake.random_int(1000, 9999),
            'response_time': random.expovariate(1/150)
        }
        
        if log['level'] == 'ERROR':
            log['message'] = random.choice([
                'Database connection timeout',
                'OutOfMemoryError: Java heap space',
                'NullPointerException in service layer'
            ])
            self.metrics['java_errors'].inc()
        elif log['level'] == 'WARN':
            log['message'] = random.choice([
                'High memory usage detected',
                'Slow database queries',
                'Connection pool at 80% capacity'
            ])
        else:
            log['message'] = f'Request processed successfully for user {log["user_id"]}'
            
        return log
    
    def _generate_k8s_logs(self):
        log = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'kubernetes',
            'level': random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.75, 0.15, 0.1])[0],
            'message': '',
            'pod_name': f'pod-{fake.random_int(1, 100)}',
            'namespace': 'production'
        }
        
        if log['level'] == 'ERROR':
            log['message'] = random.choice([
                'Pod crashed due to OOM',
                'Image pull failed',
                'Node not ready'
            ])
        elif log['level'] == 'WARN':
            log['message'] = random.choice([
                'High memory pressure on node',
                'CPU throttling detected',
                'Network latency increased'
            ])
        else:
            log['message'] = 'Pod started successfully'
            
        return log
    
    def _generate_cobol_logs(self):
        log = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': 'cobol_mainframe',
            'level': random.choices(['INFO', 'WARN', 'ERROR'], weights=[0.8, 0.15, 0.05])[0],
            'message': '',
            'job_id': f'JOB{fake.random_int(10000, 99999)}',
            'region': 'MAINFRAME_PROD'
        }
        
        if log['level'] == 'ERROR':
            log['message'] = random.choice([
                'ABEND: U4038 System error',
                'Dataset mount failed',
                'CICS transaction abended'
            ])
            self.metrics['cobol_errors'].inc()
        elif log['level'] == 'WARN':
            log['message'] = random.choice([
                'Batch job running longer than expected',
                'Storage allocation nearing limit',
                'Tape drive response slow'
            ])
        else:
            log['message'] = 'Batch job completed successfully'
            self.metrics['cobol_jobs'].inc()
            
        return log
    
    def generate_metrics(self):
        # Update system metrics
        self.metrics['java_cpu'].set(random.uniform(20, 95))
        self.metrics['java_memory'].set(random.uniform(30, 90))
        self.metrics['k8s_cpu'].set(random.uniform(15, 85))
        self.metrics['k8s_memory'].set(random.uniform(25, 80))
        self.metrics['k8s_pods'].set(random.randint(5, 50))
        self.metrics['cobol_cpu'].set(random.uniform(10, 70))
    
    def run(self):
        start_http_server(8000)
        print("Starting enhanced synthetic data generation...")
        
        while True:
            # Generate logs for all systems
            for system_name, generator in self.systems.items():
                log = generator()
                print(json.dumps(log))
            
            # Update metrics
            self.generate_metrics()
            
            time.sleep(2)

if __name__ == "__main__":
    generator = SyntheticSystemLogs()
    generator.run()