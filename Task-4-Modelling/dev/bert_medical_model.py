#!/usr/bin/env python3
"""
Medical Advisor Model using Bio_ClinicalBERT from Hugging Face
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re
import time
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class BERTMedicalModel:
    """
    A medical condition matcher using Bio_ClinicalBERT embeddings
    for semantic matching of symptoms and diagnoses.
    """
    
    def __init__(self, data_path, device=None):
        """
        Initialize the model with the given dataset path
        
        Args:
            data_path (str): Path to the CSV file containing medical data
            device (str, optional): Device to run the model on ('cuda', 'cpu', etc.)
        """
        self.data_path = data_path
        self.df = None
        
        # Set device (GPU if available, otherwise CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load Bio_ClinicalBERT model and tokenizer
        self.model_name = "emilyalsentzer/Bio_ClinicalBERT"
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        # Load and preprocess data
        self.embeddings = None
        self.load_data()
    
    def _preprocess_text(self, text):
        """Clean and normalize text for better matching"""
        if pd.isna(text) or text == "":
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self):
        """Load and preprocess the medical data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Preprocess text columns
            for col in ['Symptoms', 'Diagnosis', 'Disease', 'Treatment']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(self._preprocess_text)
            
            # Combine symptoms and diagnosis for embedding
            self.df['Combined'] = self.df['Symptoms'] + ' [SEP] ' + self.df['Diagnosis']
            
            # Generate embeddings for all conditions
            print("Generating embeddings for all medical conditions...")
            self.embeddings = self._generate_embeddings(self.df['Combined'].tolist())
            
            print(f"Loaded and preprocessed {len(self.df)} medical conditions")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def _get_bert_embedding(self, text):
        """
        Generate BERT embeddings for a single text
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Add special tokens and prepare for the model
        encoded_input = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Use the [CLS] token embedding as the sentence embedding
        # This is the first token in the sequence (index 0)
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings[0]
    
    def _generate_embeddings(self, texts):
        """
        Generate BERT embeddings for a list of texts
        
        Args:
            texts (list): List of texts to generate embeddings for
            
        Returns:
            numpy.ndarray: Matrix of embedding vectors
        """
        embeddings = []
        for text in tqdm(texts):
            embedding = self._get_bert_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _extract_symptom_tags(self, symptom_text, max_tags=5):
        """Extract key symptoms as tags from text"""
        if not symptom_text:
            return []
        
        # Split by common separators and take the first few
        separators = [';', ',', '.', 'and']
        
        for sep in separators:
            if sep in symptom_text:
                parts = [p.strip() for p in symptom_text.split(sep) if p.strip()]
                if parts:
                    return parts[:max_tags]
        
        # If no separators found, just return the whole text as one tag
        return [symptom_text]
    
    def predict(self, symptoms, diagnosis=None, top_n=3, min_similarity=0.5):
        """
        Match symptoms and diagnosis to potential conditions using BERT embeddings
        
        Args:
            symptoms (str): Description of symptoms
            diagnosis (str, optional): Any diagnostic information
            top_n (int): Number of top matches to return
            min_similarity (float): Minimum similarity threshold (0-1)
            
        Returns:
            list: List of matching conditions with scores and details
        """
        if self.df is None or self.df.empty or self.embeddings is None:
            return []
        
        # Preprocess input
        symptoms = self._preprocess_text(symptoms)
        diagnosis = self._preprocess_text(diagnosis) if diagnosis else ""
        
        # Combine input
        combined_input = symptoms + " [SEP] " + diagnosis
        
        # Generate embedding for input
        input_embedding = self._get_bert_embedding(combined_input)
        
        # Calculate similarity with all conditions
        similarities = cosine_similarity([input_embedding], self.embeddings)[0]
        
        # Get the indices of the top N similar diseases
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Create result list
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score >= min_similarity:  # Only include if above threshold
                # Convert similarity to percentage for consistency with other models
                similarity_percentage = int(similarity_score * 100)
                
                # Create a symptom tag list
                symptom_tags = self._extract_symptom_tags(self.df.iloc[idx]['Symptoms'])
                
                # Calculate match quality
                match_quality = "low"
                if similarity_percentage >= 70:
                    match_quality = "high"
                elif similarity_percentage >= 50:
                    match_quality = "medium"
                
                results.append({
                    'disease': self.df.iloc[idx]['Disease'],
                    'match_percentage': similarity_percentage,
                    'match_quality': match_quality,
                    'treatment': self.df.iloc[idx]['Treatment'],
                    'symptoms': self.df.iloc[idx]['Symptoms'],
                    'symptom_tags': symptom_tags,
                    'diagnosis': self.df.iloc[idx]['Diagnosis'],
                    'similarity_score': similarity_score,
                    'layman_terms': self.df.iloc[idx]['Laymen Terms'] if 'Laymen Terms' in self.df.columns else ''
                })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = BERTMedicalModel("combined_diseases_v2.csv")
    
    # Test with a few examples
    test_cases = [
        {"symptoms": "Persistent cough, fever, fatigue, shortness of breath", 
         "diagnosis": "Chest X-ray shows infiltrates"},
        {"symptoms": "Headache, nausea, sensitivity to light", 
         "diagnosis": ""},
        {"symptoms": "Frequent urination, increased thirst, unexplained weight loss", 
         "diagnosis": "Blood glucose level 240 mg/dL"}
    ]
    
    print("\nTesting Bio_ClinicalBERT Medical Model:")
    print("=====================================")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Symptoms: {case['symptoms']}")
        if case['diagnosis']:
            print(f"Diagnosis: {case['diagnosis']}")
        
        start_time = time.time()
        results = model.predict(case['symptoms'], case['diagnosis'])
        inference_time = time.time() - start_time
        
        print(f"\nTop matches (inference time: {inference_time:.2f}s):")
        for j, result in enumerate(results):
            print(f"{j+1}. {result['disease']} ({result['match_percentage']}% match, {result['match_quality']} quality)")
            print(f"   Key symptoms: {', '.join(result['symptom_tags'])}")
            print(f"   Treatment: {result['treatment'][:100]}...")
