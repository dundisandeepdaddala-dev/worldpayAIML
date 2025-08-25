# rca/llm_analyzer.py
import requests
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

class LLMAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = faiss.IndexFlatL2(384)  # Dimension of MiniLM embeddings
        self.log_db = []
    
    def embed_text(self, text):
        """Generate embeddings for text"""
        return self.embedding_model.encode([text])[0]
    
    def add_to_vector_db(self, log_entry):
        """Add log entry to vector database"""
        if 'message' not in log_entry:
            return
        
        embedding = self.embed_text(log_entry['message'])
        self.vector_db.add(np.array([embedding], dtype=np.float32))
        self.log_db.append(log_entry)
    
    def search_similar_logs(self, query, k=5):
        """Search for similar logs in vector database"""
        query_embedding = self.embed_text(query)
        distances, indices = self.vector_db.search(np.array([query_embedding], dtype=np.float32), k)
        
        similar_logs = []
        for idx in indices[0]:
            if idx < len(self.log_db):
                similar_logs.append(self.log_db[idx])
        
        return similar_logs
    
    def generate_llm_analysis(self, prompt):
        """Generate analysis using LLM"""
        try:
            data = {
                "model": "llama3.1",
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=data)
            response.raise_for_status()
            result = response.json()
            
            return result['response']
        except Exception as e:
            return f"Error querying LLM: {e}"
    
    def perform_rca(self, anomaly_info, related_logs, metrics_data):
        """Perform root cause analysis using LLM and RAG"""
        # Create context from similar logs and metrics
        context = "Related logs from vector database:\n"
        for i, log in enumerate(related_logs[:3]):  # Top 3 similar logs
            context += f"{i+1}. {log['timestamp']} - {log['system']} - {log['level']}: {log['message']}\n"
        
        context += f"\nMetrics data at time of anomaly:\n{json.dumps(metrics_data, indent=2)}"
        
        prompt = f"""
        Perform root cause analysis for the following system anomaly:
        
        Anomaly Details:
        - System: {anomaly_info['system']}
        - Time: {anomaly_info['timestamp']}
        - Metric: {anomaly_info['metric']}
        - Value: {anomaly_info['value']}
        - Expected Range: {anomaly_info['expected_range']}
        
        Context from similar historical incidents:
        {context}
        
        Please provide a comprehensive analysis with:
        1. Root cause hypothesis
        2. Evidence supporting this hypothesis
        3. Recommended immediate actions
        4. Long-term prevention strategies
        
        Format your response clearly with headings.
        """
        
        return self.generate_llm_analysis(prompt)
    
    def generate_shap_analysis(self, model, features, feature_names):
        """Generate SHAP analysis for model explainability"""
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            
            # Create summary plot
            shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
            
            return "SHAP analysis completed. Feature importance visualized."
        except Exception as e:
            return f"SHAP analysis error: {e}"