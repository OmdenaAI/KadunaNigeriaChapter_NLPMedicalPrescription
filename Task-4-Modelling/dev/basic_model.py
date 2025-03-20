#!/usr/bin/env python3
"""
Basic Medical Advisor Model using simple TF-IDF matching
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BasicMedicalModel:
    """
    A simple medical condition matcher using TF-IDF vectorization
    and cosine similarity for basic symptom matching.
    """
    
    def __init__(self, data_path):
        """Initialize the model with the given dataset path"""
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the medical data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} medical conditions")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def predict(self, symptoms, diagnosis=None, top_n=3):
        """
        Match symptoms and diagnosis to potential conditions
        
        Args:
            symptoms (str): Description of symptoms
            diagnosis (str, optional): Any diagnostic information
            top_n (int): Number of top matches to return
            
        Returns:
            list: List of matching conditions with scores
        """
        if self.df is None or self.df.empty:
            return []
        
        # Create a working copy of the dataframe
        df = self.df.copy()
        
        # Combine symptoms and diagnosis for comparison
        user_input = f"{symptoms} {diagnosis}" if diagnosis else symptoms
        
        # Prepare data for comparison
        df['Combined'] = df['Symptoms'] + ' ' + df['Diagnosis']
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Combined'].tolist() + [user_input])
        
        # Calculate similarity
        user_vector = tfidf_matrix[-1]
        content_vectors = tfidf_matrix[:-1]
        similarity_scores = cosine_similarity(user_vector, content_vectors).flatten()
        
        # Get the indices of the top N similar diseases
        top_indices = similarity_scores.argsort()[:-top_n-1:-1]
        
        # Create result list
        results = []
        for idx in top_indices:
            similarity_percentage = int(similarity_scores[idx] * 100)
            if similarity_percentage > 0:  # Only include if there's some similarity
                results.append({
                    'disease': df.iloc[idx]['Disease'],
                    'match_percentage': similarity_percentage,
                    'treatment': df.iloc[idx]['Treatment'],
                    'symptoms': df.iloc[idx]['Symptoms'],
                    'diagnosis': df.iloc[idx]['Diagnosis'],
                    'layman_terms': df.iloc[idx]['Laymen Terms'] if 'Laymen Terms' in df.columns else ''
                })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = BasicMedicalModel("combined_diseases_v2.csv")
    
    # Test with a few examples
    test_cases = [
        {"symptoms": "Persistent cough, fever, fatigue, shortness of breath", 
         "diagnosis": "Chest X-ray shows infiltrates"},
        {"symptoms": "Headache, nausea, sensitivity to light", 
         "diagnosis": ""},
        {"symptoms": "Frequent urination, increased thirst, unexplained weight loss", 
         "diagnosis": "Blood glucose level 240 mg/dL"}
    ]
    
    print("\nTesting Basic Medical Model:")
    print("============================")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Symptoms: {case['symptoms']}")
        if case['diagnosis']:
            print(f"Diagnosis: {case['diagnosis']}")
        
        results = model.predict(case['symptoms'], case['diagnosis'])
        
        print(f"\nTop matches:")
        for j, result in enumerate(results):
            print(f"{j+1}. {result['disease']} ({result['match_percentage']}% match)")
            print(f"   Treatment: {result['treatment'][:100]}...")
