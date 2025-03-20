#!/usr/bin/env python3
"""
Improved Medical Advisor Model using enhanced TF-IDF matching
with symptom weighting and bigram analysis
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class ImprovedMedicalModel:
    """
    An enhanced medical condition matcher using weighted TF-IDF vectorization,
    bigram analysis, and improved text preprocessing for better symptom matching.
    """
    
    def __init__(self, data_path):
        """Initialize the model with the given dataset path"""
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the medical data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Preprocess text columns
            for col in ['Symptoms', 'Diagnosis', 'Disease', 'Treatment']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(self._preprocess_text)
                    
            print(f"Loaded and preprocessed {len(self.df)} medical conditions")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
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
    
    def predict(self, symptoms, diagnosis=None, top_n=3, min_similarity=15):
        """
        Match symptoms and diagnosis to potential conditions with enhanced matching
        
        Args:
            symptoms (str): Description of symptoms
            diagnosis (str, optional): Any diagnostic information
            top_n (int): Number of top matches to return
            min_similarity (int): Minimum similarity threshold (%)
            
        Returns:
            list: List of matching conditions with scores and details
        """
        if self.df is None or self.df.empty:
            return []
        
        # Create a working copy of the dataframe
        df = self.df.copy()
        
        # Preprocess user input
        symptoms = self._preprocess_text(symptoms)
        diagnosis = self._preprocess_text(diagnosis) if diagnosis else ""
        
        # Prepare data for comparison
        # Weight symptoms more heavily than diagnosis by duplicating them
        df['SearchText'] = df['Symptoms'] + ' ' + df['Symptoms'] + ' ' + df['Diagnosis']
        
        # Combine user input (weighting symptoms more)
        user_input = ''
        if symptoms:
            user_input += symptoms + ' ' + symptoms  # Duplicate to give more weight
        if diagnosis:
            user_input += ' ' + diagnosis
            
        if not user_input.strip():
            return []
        
        # Create TF-IDF vectors with better parameters
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            min_df=2,            # Ignore very rare terms
            max_df=0.9           # Ignore very common terms
        )
        
        try:
            # Calculate similarity
            tfidf_matrix = vectorizer.fit_transform(df['SearchText'].tolist() + [user_input])
            user_vector = tfidf_matrix[-1]
            content_vectors = tfidf_matrix[:-1]
            similarity_scores = cosine_similarity(user_vector, content_vectors).flatten()
            
            # Get the indices of the top N similar diseases
            top_indices = similarity_scores.argsort()[:-top_n-1:-1]
            
            # Create result list
            results = []
            for idx in top_indices:
                similarity_percentage = int(similarity_scores[idx] * 100)
                if similarity_percentage >= min_similarity:  # Only include if above threshold
                    # Create a symptom tag list
                    symptom_tags = self._extract_symptom_tags(df.iloc[idx]['Symptoms'])
                    
                    # Calculate match quality
                    match_quality = "low"
                    if similarity_percentage >= 50:
                        match_quality = "high"
                    elif similarity_percentage >= 25:
                        match_quality = "medium"
                    
                    results.append({
                        'disease': df.iloc[idx]['Disease'],
                        'match_percentage': similarity_percentage,
                        'match_quality': match_quality,
                        'treatment': df.iloc[idx]['Treatment'],
                        'symptoms': df.iloc[idx]['Symptoms'],
                        'symptom_tags': symptom_tags,
                        'diagnosis': df.iloc[idx]['Diagnosis'],
                        'layman_terms': df.iloc[idx]['Laymen Terms'] if 'Laymen Terms' in df.columns else ''
                    })
            
            return results
        except Exception as e:
            print(f"Error in search: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = ImprovedMedicalModel("../data/combined_diseases_v2.csv")
    
    # Test with a few examples
    test_cases = [
        {"symptoms": "Persistent cough, fever, fatigue, shortness of breath", 
         "diagnosis": "Chest X-ray shows infiltrates"},
        {"symptoms": "Headache, nausea, sensitivity to light", 
         "diagnosis": ""},
        {"symptoms": "Frequent urination, increased thirst, unexplained weight loss", 
         "diagnosis": "Blood glucose level 240 mg/dL"}
    ]
    
    print("\nTesting Improved Medical Model:")
    print("==============================")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Symptoms: {case['symptoms']}")
        if case['diagnosis']:
            print(f"Diagnosis: {case['diagnosis']}")
        
        results = model.predict(case['symptoms'], case['diagnosis'])
        
        print(f"\nTop matches:")
        for j, result in enumerate(results):
            print(f"{j+1}. {result['disease']} ({result['match_percentage']}% match, {result['match_quality']} quality)")
            print(f"   Key symptoms: {', '.join(result['symptom_tags'])}")
            print(f"   Treatment: {result['treatment'][:100]}...")
