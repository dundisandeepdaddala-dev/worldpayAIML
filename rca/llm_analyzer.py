# rca/llm_analyzer.py
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import time

class LLMAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.log_db = []
        self.ollama_available = self._check_ollama_availability()
        
    def _check_ollama_availability(self):
        """Check if Ollama is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            print("⚠️  Ollama not available. Using fallback analysis.")
            return False
    
    def add_to_vector_db(self, log_entry):
        """Add log entry to database"""
        self.log_db.append(log_entry)
    
    def search_similar_logs(self, query, k=5):
        """Search for similar logs in database"""
        # Simple keyword-based search for demo
        similar_logs = []
        query_words = query.lower().split()
        
        for log in self.log_db:
            if any(word in log.get('message', '').lower() for word in query_words):
                similar_logs.append(log)
            if len(similar_logs) >= k:
                break
                
        return similar_logs
    
    def generate_llm_analysis(self, prompt, max_retries=2):
        """Generate analysis using LLM with retry logic"""
        if not self.ollama_available:
            return self._generate_fallback_analysis(prompt)
            
        for attempt in range(max_retries):
            try:
                data = {
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for more deterministic responses
                        "top_p": 0.9,
                        "num_predict": 300  # Limit response length
                    }
                }
                
                response = requests.post(self.ollama_url, json=data, timeout=15)
                response.raise_for_status()
                result = response.json()
                
                return result['response']
            except requests.exceptions.Timeout:
                print(f"LLM timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"LLM error: {e}")
                self.ollama_available = False
                return self._generate_fallback_analysis(prompt)
                
        print("LLM timeout after retries, using fallback")
        return self._generate_fallback_analysis(prompt)
    
    def _generate_fallback_analysis(self, prompt):
        """Generate fallback analysis when LLM is unavailable"""
        # Extract key information from the prompt for fallback analysis
        if "java" in prompt.lower():
            return """FALLBACK RCA FOR JAVA APPLICATION:
ROOT CAUSE: NullPointerException typically indicates missing object initialization or null references.
IMMEDIATE ACTION: Check recent code deployments, review stack traces for the exact location of the NPE.
LONG-TERM: Implement better null checking and use Optional types where appropriate."""
        
        elif "kubernetes" in prompt.lower() or "k8s" in prompt.lower():
            return """FALLBACK RCA FOR KUBERNETES:
ROOT CAUSE: Pod crashes often relate to resource constraints (CPU/Memory) or misconfigured deployments.
IMMEDIATE ACTION: Check pod logs with kubectl, verify resource requests/limits.
LONG-TERM: Implement resource monitoring and autoscaling."""
        
        elif "cobol" in prompt.lower():
            return """FALLBACK RCA FOR COBOL SYSTEM:
ROOT CAUSE: Storage allocation issues typically indicate dataset space exhaustion or inefficient job processing.
IMMEDIATE ACTION: Check dataset allocations, verify job scheduling.
LONG-TERM: Implement storage monitoring and cleanup procedures."""
        
        elif "metric" in prompt.lower():
            return """FALLBACK RCA FOR METRIC ANOMALY:
ROOT CAUSE: Abnormal metric values often indicate resource exhaustion, performance degradation, or system overload.
IMMEDIATE ACTION: Check system resources, review recent changes or deployments.
LONG-TERM: Implement better capacity planning and auto-scaling."""
        
        else:
            return """FALLBACK RCA:
ROOT CAUSE: Unknown system issue detected.
IMMEDIATE ACTION: Review logs and metrics for unusual patterns.
LONG-TERM: Enhance monitoring and alerting for similar issues."""
    
    def perform_rca(self, anomaly_info, related_logs, metrics_data):
        """Perform root cause analysis using LLM with fallback"""
        # Create context from similar logs
        context = "Related historical incidents:\n"
        for i, log in enumerate(related_logs[:3]):  # Top 3 similar logs
            context += f"{i+1}. {log.get('timestamp', '')} - {log.get('system', '')} - {log.get('level', '')}: {log.get('message', '')}\n"
        
        # Create a more structured prompt with clear instructions
        prompt = f"""
        You are an expert Site Reliability Engineer (SRE) performing root cause analysis.
        
        Analyze this anomaly and provide a structured response with:
        1. ROOT CAUSE: [Brief explanation of the likely root cause]
        2. IMMEDIATE ACTION: [One specific action to take immediately]
        3. LONG-TERM FIX: [One recommendation to prevent recurrence]
        
        ANOMALY DETAILS:
        - System: {anomaly_info.get('system', 'unknown')}
        - Time: {anomaly_info.get('timestamp', 'unknown')}
        - Type: {anomaly_info.get('type', 'unknown')}
        - Message: {anomaly_info.get('message', 'unknown')}
        - Severity Score: {anomaly_info.get('score', 'unknown')}/10
        
        CONTEXT FROM SIMILAR INCIDENTS:
        {context}
        
        Please provide a concise, structured response.
        """
        
        return self.generate_llm_analysis(prompt)