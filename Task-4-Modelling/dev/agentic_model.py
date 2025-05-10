#!/usr/bin/env python3
"""
Agentic Medical Advisor Model combining database search with LLM reasoning
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
import http.client
import urllib3

# Load environment variables (including API keys)
load_dotenv()

class AgenticMedicalModel:
    """
    An advanced medical condition matcher that combines database search
    with LLM reasoning to provide improved symptom-to-disease matching.
    """
    
    def __init__(self, data_path, use_mock_llm=False):
        """
        Initialize the model with the given dataset path
        
        Args:
            data_path (str): Path to the CSV file containing medical data
            use_mock_llm (bool): If True, use a simulated LLM instead of API call
        """
        self.data_path = data_path
        self.df = None
        self.use_mock_llm = use_mock_llm
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_data()
        
        # Configure the OpenAI client if not using mock
        if not use_mock_llm:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: No OpenAI API key found. Set OPENAI_API_KEY in environment or .env file.")
                print("Falling back to mock LLM.")
                self.use_mock_llm = True
            else:
                # Create OpenAI client without proxy settings
                # In newer versions, proxy settings should be handled through environment variables
                try:
                    self.client = OpenAI(api_key=api_key)
                except TypeError as e:
                    # Handle the case when proxies might be passed through environment
                    print(f"Warning: {e}")
                    print("Attempting to initialize without proxy settings...")
                    # Clean environment proxy variables that might affect the client
                    proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
                    saved_proxies = {}
                    for var in proxy_env_vars:
                        if var in os.environ:
                            saved_proxies[var] = os.environ[var]
                            os.environ.pop(var)
                    
                    # Try creating client again
                    try:
                        self.client = OpenAI(api_key=api_key)
                        print("Successfully initialized OpenAI client without proxy settings.")
                    except Exception as e2:
                        print(f"Error initializing OpenAI client: {e2}")
                        print("Falling back to mock LLM.")
                        self.use_mock_llm = True
                    
                    # Restore environment variables
                    for var, value in saved_proxies.items():
                        os.environ[var] = value
                except Exception as e:
                    print(f"Error initializing OpenAI client: {e}")
                    print("Falling back to mock LLM.")
                    self.use_mock_llm = True
    
    def load_data(self):
        """Load and preprocess the medical data from CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Preprocess text columns
            for col in ['Symptoms', 'Diagnosis', 'Disease', 'Treatment']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].apply(self._preprocess_text)
            
            # Prepare vector search
            self._preprocess_data()
                    
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
    
    def _preprocess_data(self):
        """Preprocess the medical data for vector search"""
        # Combine symptoms and diagnosis for comparison
        self.df['Combined'] = self.df['Symptoms'] + ' ' + self.df['Diagnosis']
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Combined'].tolist())
    
    def _search_database(self, symptoms, diagnosis, top_n=5):
        """
        Search the database using vector similarity
        
        Args:
            symptoms (str): Preprocessed symptom text
            diagnosis (str): Preprocessed diagnosis text
            top_n (int): Number of top matches to return
            
        Returns:
            list: List of top matching conditions
        """
        # Combine user input with symptom weighting
        user_input = ''
        if symptoms:
            user_input += symptoms + ' ' + symptoms  # Duplicate to give more weight
        if diagnosis:
            user_input += ' ' + diagnosis
            
        if not user_input.strip():
            return []
        
        # Create a vector for the user query
        query_vector = self.vectorizer.transform([user_input])
        
        # Calculate similarity
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get the indices of the top N similar diseases
        top_indices = similarity_scores.argsort()[:-top_n-1:-1]
        
        # Create result list
        results = []
        for idx in top_indices:
            similarity_percentage = int(similarity_scores[idx] * 100)
            if similarity_percentage > 10:  # Only include if there's meaningful similarity
                results.append({
                    'disease': self.df.iloc[idx]['Disease'],
                    'match_percentage': similarity_percentage,
                    'treatment': self.df.iloc[idx]['Treatment'],
                    'symptoms': self.df.iloc[idx]['Symptoms'],
                    'diagnosis': self.df.iloc[idx]['Diagnosis'],
                    'layman_terms': self.df.iloc[idx]['Laymen Terms'] if 'Laymen Terms' in self.df.columns else ''
                })
        
        return results
    
    def _analyze_with_llm(self, symptoms, diagnosis, database_results):
        """
        Use LLM to analyze symptoms and database results
        
        Args:
            symptoms (str): Raw symptom text
            diagnosis (str): Raw diagnosis text
            database_results (list): Results from database search
            
        Returns:
            dict: LLM analysis with likely conditions and advice
        """
        # Create a prompt for the LLM
        database_context = ""
        for i, result in enumerate(database_results[:3]):
            database_context += f"""
            Result {i+1}:
            Disease: {result['disease']}
            Match: {result['match_percentage']}%
            Symptoms: {result['symptoms']}
            Diagnosis method: {result['diagnosis']}
            Treatment: {result['treatment']}
            """
        
        prompt = f"""
        I need to analyze the following medical case:
        
        Symptoms: {symptoms}
        Diagnostic Information: {diagnosis}
        
        Here are the top matches from our medical database:
        {database_context}
        
        Please analyze this information and provide:
        1. Your assessment of the most likely conditions based on the symptoms and diagnostic info
        2. How well they match the database results
        3. Any additional conditions that should be considered but weren't in the database results
        4. General advice for the patient
        
        Respond in JSON format with these keys: 
        - likely_conditions (array of objects with name, confidence_percentage, and reasoning)
        - additional_considerations (array of condition names that weren't in database but should be considered)
        - patient_advice (string with general advice)
        """
        
        # If using mock LLM, simulate a response
        if self.use_mock_llm:
            return self._mock_llm_response(symptoms, diagnosis, database_results)
        
        # Otherwise call OpenAI API using new client syntax
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or use "gpt-3.5-turbo" for lower cost
                messages=[
                    {"role": "system", "content": "You are an AI medical assistant working with a database search system. Your task is to analyze symptoms and diagnostic information and provide helpful insights. You should NOT diagnose but provide information about potential matches and considerations. Always emphasize the importance of seeking professional medical advice."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more focused responses
            )
            
            # Extract the response
            content = response.choices[0].message.content
            
            # Try to parse JSON from the response
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                    analysis = json.loads(json_str)
                else:
                    # Try to parse the entire content
                    analysis = json.loads(content)
            except:
                # Fallback if JSON parsing fails
                print("Failed to parse LLM response. Using database results only.")
                analysis = {
                    "likely_conditions": [
                        {"name": result["disease"], 
                         "confidence_percentage": result["match_percentage"], 
                         "reasoning": "Based on database match."}
                        for result in database_results[:3]
                    ],
                    "additional_considerations": [],
                    "patient_advice": "Please consult a healthcare professional for proper diagnosis and treatment."
                }
            
            return analysis
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            # Return simplified results based on database search
            return {
                "likely_conditions": [
                    {"name": result["disease"], 
                     "confidence_percentage": result["match_percentage"], 
                     "reasoning": "Based on database match."}
                    for result in database_results[:3]
                ],
                "additional_considerations": [],
                "patient_advice": "Please consult a healthcare professional for proper diagnosis and treatment."
            }
    
    def _mock_llm_response(self, symptoms, diagnosis, database_results):
        """
        Create a simulated LLM response based on database results
        
        This allows the agentic model to work without an API key
        """
        likely_conditions = []
        
        for i, result in enumerate(database_results[:3]):
            # Apply some transformations to make the mock response different from pure database results
            confidence = result['match_percentage']
            
            # Adjust confidence based on symptom and diagnosis overlap
            symptom_terms = set(result['symptoms'].split())
            user_symptom_terms = set(symptoms.lower().split())
            overlap = len(symptom_terms.intersection(user_symptom_terms))
            
            # Small adjustments to confidence
            if overlap > 5:
                confidence += 5
            elif overlap < 2:
                confidence -= 5
                
            # Cap confidence at 98%
            confidence = min(98, max(10, confidence))
            
            reasoning = f"This condition matches many of the described symptoms"
            if diagnosis and result['diagnosis']:
                if any(term in diagnosis.lower() for term in result['diagnosis'].lower().split()):
                    reasoning += " and is consistent with the diagnostic information"
                    confidence += 3
            
            likely_conditions.append({
                "name": result['disease'],
                "confidence_percentage": confidence,
                "reasoning": reasoning
            })
        
        # Generate some additional considerations
        additional_considerations = []
        if "fever" in symptoms.lower() and "cough" in symptoms.lower():
            additional_considerations.append("Influenza")
        if "headache" in symptoms.lower() and "pain" in symptoms.lower():
            additional_considerations.append("Tension Headache")
        
        # Generate advice
        advice = "Based on the symptoms described, it would be advisable to consult with a healthcare professional for proper evaluation."
        if any("fever" in result['symptoms'] for result in database_results[:3]):
            advice += " Monitor fever and stay hydrated."
        if any("pain" in result['symptoms'] for result in database_results[:3]):
            advice += " Rest and avoid activities that worsen symptoms."
        
        return {
            "likely_conditions": likely_conditions,
            "additional_considerations": additional_considerations,
            "patient_advice": advice
        }
    
    def _get_condition_details(self, condition_name):
        """
        Get details for a specific condition from the database
        
        Args:
            condition_name (str): Name of the condition to look up
            
        Returns:
            dict: Condition details or None if not found
        """
        matches = self.df[self.df['Disease'].str.contains(condition_name, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None
    
    def predict(self, symptoms, diagnosis=None, top_n=3):
        """
        Process a medical case to find matching conditions
        
        Args:
            symptoms (str): Description of symptoms
            diagnosis (str, optional): Any diagnostic information
            top_n (int): Number of top matches to return
            
        Returns:
            dict: Structured results with database matches and LLM analysis
        """
        # Preprocess inputs
        symptoms_processed = self._preprocess_text(symptoms)
        diagnosis_processed = self._preprocess_text(diagnosis) if diagnosis else ""
        
        # Step 1: Search the database
        database_results = self._search_database(symptoms_processed, diagnosis_processed)
        
        # Step 2: Analyze with LLM
        llm_analysis = self._analyze_with_llm(symptoms, diagnosis if diagnosis else "", database_results)
        
        # Step 3: Combine results
        final_results = []
        
        # Process LLM suggestions first
        for condition in llm_analysis["likely_conditions"]:
            result_item = {
                "disease": condition["name"],
                "match_percentage": condition["confidence_percentage"],
                "reasoning": condition["reasoning"],
                "source": "llm"
            }
            
            # Try to find detailed info in database
            for db_match in database_results:
                if condition["name"].lower() in db_match["disease"].lower():
                    result_item.update({
                        "treatment": db_match["treatment"],
                        "symptoms": db_match["symptoms"],
                        "diagnosis": db_match["diagnosis"],
                        "layman_terms": db_match["layman_terms"] if "layman_terms" in db_match else ""
                    })
                    break
            
            final_results.append(result_item)
        
        # Add database results that weren't in LLM suggestions
        llm_diseases = [cond["name"].lower() for cond in llm_analysis["likely_conditions"]]
        for db_match in database_results:
            if not any(db_match["disease"].lower() in disease for disease in llm_diseases):
                final_results.append({
                    "disease": db_match["disease"],
                    "match_percentage": db_match["match_percentage"],
                    "treatment": db_match["treatment"],
                    "symptoms": db_match["symptoms"],
                    "diagnosis": db_match["diagnosis"],
                    "layman_terms": db_match["layman_terms"] if "layman_terms" in db_match else "",
                    "source": "database"
                })
        
        # Sort by match percentage and limit to top_n
        final_results = sorted(final_results, key=lambda x: x["match_percentage"], reverse=True)[:top_n]
        
        return {
            "conditions": final_results,
            "additional_considerations": llm_analysis.get("additional_considerations", []),
            "patient_advice": llm_analysis.get("patient_advice", "Please consult a healthcare professional for proper diagnosis and treatment.")
        }

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = AgenticMedicalModel("combined_diseases_v2.csv", use_mock_llm=True)
    
    # Test with a few examples
    test_cases = [
        {"symptoms": "Persistent cough, fever, fatigue, shortness of breath", 
         "diagnosis": "Chest X-ray shows infiltrates"},
        {"symptoms": "Headache, nausea, sensitivity to light", 
         "diagnosis": ""},
        {"symptoms": "Frequent urination, increased thirst, unexplained weight loss", 
         "diagnosis": "Blood glucose level 240 mg/dL"}
    ]
    
    print("\nTesting Agentic Medical Model:")
    print("=============================")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Symptoms: {case['symptoms']}")
        if case['diagnosis']:
            print(f"Diagnosis: {case['diagnosis']}")
        
        results = model.predict(case['symptoms'], case['diagnosis'])
        
        print(f"\nTop matches:")
        for j, result in enumerate(results["conditions"]):
            print(f"{j+1}. {result['disease']} ({result['match_percentage']}% match)")
            if "reasoning" in result:
                print(f"   Reasoning: {result['reasoning']}")
            if "treatment" in result:
                print(f"   Treatment: {result['treatment'][:100]}...")
        
        if results["additional_considerations"]:
            print(f"\nAdditional considerations: {', '.join(results['additional_considerations'])}")
        
        print(f"\nPatient advice: {results['patient_advice']}")