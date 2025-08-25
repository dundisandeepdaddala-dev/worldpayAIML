# rca/llm_analyzer.py
import requests
import json
import numpy as np
from datetime import datetime, timedelta

class LLMAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.log_db = []
    
    def add_to_vector_db(self, log_entry):
        """Add log entry to database"""
        self.log_db.append(log_entry)
    
    def search_similar_logs(self, query, k=5):
        """Search for similar logs in database"""
        # Simple keyword-based search for demo
        similar_logs = []
        for log in self.log_db:
            if any(word in log.get('message', '').lower() for word in query.lower().split()):
                similar_logs.append(log)
            if len(similar_logs) >= k:
                break
        return similar_logs
    
    def generate_llm_analysis(self, prompt):
        """Generate analysis using LLM"""
        try:
            data = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            return result['response']
        except Exception as e:
            return f"Error querying LLM: {e}"
    
    def perform_rca(self, anomaly_info, related_logs, metrics_data):
        """Perform root cause analysis using LLM"""
        # Create context from similar logs
        context = "Related logs from database:\n"
        for i, log in enumerate(related_logs[:3]):  # Top 3 similar logs
            context += f"{i+1}. {log.get('timestamp', '')} - {log.get('system', '')} - {log.get('level', '')}: {log.get('message', '')}\n"
        
        prompt = f"""
        Perform root cause analysis for the following system anomaly:
        
        Anomaly Details:
        - System: {anomaly_info.get('system', 'unknown')}
        - Time: {anomaly_info.get('timestamp', 'unknown')}
        - Type: {anomaly_info.get('type', 'unknown')}
        - Message: {anomaly_info.get('message', 'unknown')}
        
        Context from similar historical incidents:
        {context}
        
        Please provide a concise analysis with:
        1. Root cause hypothesis
        2. Recommended immediate actions
        
        Format your response clearly.
        """
        
        return self.generate_llm_analysis(prompt)