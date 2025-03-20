import streamlit as st
import pandas as pd
import openai
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(
    page_title="Medical Advisor",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .disclaimer {
        font-size: 12px;
        color: #6c757d;
        font-style: italic;
    }
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-text {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load the medical data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('../data/combined_diseases_v2.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Header
st.markdown("""
<div class="header-container">
    <h1>🩺 Medical Advisor</h1>
</div>
<p>Enter your symptoms and any diagnostic information to get potential matches from our medical database.</p>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <p><strong>Disclaimer:</strong> This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Input form
st.sidebar.title("Patient Information")

with st.sidebar.form("patient_info_form"):
    symptoms = st.text_area("Describe your symptoms in detail:", 
                           help="Example: Persistent cough, fever, fatigue, and shortness of breath")
    
    diagnosis_info = st.text_area("Any diagnostic information (if available):", 
                                 help="Example: X-ray showed lung infiltrates, elevated white blood cell count")
    
    severity = st.slider("Rate the severity of your symptoms:", 1, 10, 5)
    
    duration = st.selectbox("How long have you been experiencing these symptoms?", 
                           ["Less than a day", "1-3 days", "4-7 days", "1-2 weeks", 
                            "2-4 weeks", "1-3 months", "More than 3 months"])
    
    age = st.number_input("Age:", 0, 120, 30)
    
    gender = st.radio("Gender:", ["Male", "Female", "Other"])
    
    submit_button = st.form_submit_button("Get Medical Advice")

# Main content area
def query_medical_database(symptoms, diagnosis, medical_data):
    # Create a prompt for the LLM
    prompt = f"""
    I need to find potential matches for the following medical case:
    
    Symptoms: {symptoms}
    Diagnostic Information: {diagnosis}
    
    Based on this information, analyze the medical database and return the top 3 potential matches.
    For each match, provide:
    1. The disease name
    2. How well it matches the symptoms and diagnosis (on a scale of 0-100%)
    3. The recommended treatment
    4. Brief explanation in simple terms
    
    Format the response as a JSON list with these keys: disease, match_percentage, treatment, explanation
    """
    
    try:
        # Call OpenAI API with a System role and User message
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or other appropriate model
            messages=[
                {"role": "system", "content": "You are an AI medical assistant. Your task is to analyze symptoms and diagnostic information against a medical database and find potential matches. You should NOT diagnose but only provide information about potential matches. Always include a disclaimer about seeking professional medical advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        # Extract and process the response
        content = response.choices[0].message.content
        
        # For simplicity, let's assume the LLM responds with well-formatted content
        # In production, you would need more robust parsing and error handling
        import json
        try:
            # Try to extract JSON from the response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
                matches = json.loads(json_str)
            else:
                # Otherwise try to parse the entire content
                matches = json.loads(content)
        except:
            # Fallback if JSON parsing fails
            st.error("Failed to parse AI response. Please try again.")
            matches = []
        
        return matches
    except Exception as e:
        st.error(f"Error querying AI: {str(e)}")
        return []

# Agent class to handle the medical advice process
class MedicalAgent:
    def __init__(self, medical_data):
        self.medical_data = medical_data
        
    def search_database(self, symptoms, diagnosis):
        """Search the database for matching diseases based on symptoms and diagnosis"""
        # This is where we would normally do a vector search or other matching algorithm
        # Instead, we're using LLM to do the matching for demonstration purposes
        return query_medical_database(symptoms, diagnosis, self.medical_data)
    
    def get_detailed_information(self, disease_name):
        """Get detailed information about a specific disease"""
        try:
            disease_info = self.medical_data[self.medical_data['Disease'] == disease_name].iloc[0]
            return disease_info
        except:
            return None
    
    def provide_advice(self, symptoms, diagnosis):
        """Main method to provide medical advice"""
        # First, search for potential matches
        matches = self.search_database(symptoms, diagnosis)
        
        results = []
        for match in matches:
            # Get more details about each match
            disease_name = match.get('disease')
            disease_details = self.get_detailed_information(disease_name)
            
            # If found in our database, add the database information
            if disease_details is not None:
                match['database_info'] = {
                    'symptoms': disease_details['Symptoms'],
                    'diagnosis': disease_details['Diagnosis'],
                    'treatment': disease_details['Treatment'],
                    'layman_terms': disease_details['Laymen Terms']
                }
            
            results.append(match)
        
        return results

# Main application logic
def main():
    # Load medical data
    medical_data = load_data()
    
    # Initialize the medical agent
    agent = MedicalAgent(medical_data)
    
    # If form is submitted
    if submit_button:
        if not symptoms:
            st.warning("Please describe your symptoms.")
            return
        
        with st.spinner("Analyzing your information..."):
            # Simulate some processing time
            time.sleep(1)
            
            # Get advice from the agent
            results = agent.provide_advice(symptoms, diagnosis_info)
            
            if results:
                st.subheader("Potential Matches")
                st.write("Based on the information provided, these conditions might be relevant:")
                
                for i, result in enumerate(results):
                    with st.container():
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>{result.get('disease')}</h3>
                            <p><strong>Match confidence:</strong> {result.get('match_percentage', 'N/A')}</p>
                            <p><strong>Treatment:</strong> {result.get('treatment', 'Consult a healthcare professional')}</p>
                            <p><strong>Explanation:</strong> {result.get('explanation', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # If we have database info, show a detailed view
                        if 'database_info' in result:
                            with st.expander("Show medical details"):
                                st.write("**Symptoms:**", result['database_info']['symptoms'])
                                st.write("**Diagnosis:**", result['database_info']['diagnosis'])
                                st.write("**Treatment:**", result['database_info']['treatment'])
                                st.write("**In simple terms:**", result['database_info']['layman_terms'])
            else:
                st.warning("No matching conditions found. Please provide more detailed information or consult a healthcare professional.")
    
    # Reminder about seeking professional advice
    st.markdown("""
    <div class="disclaimer">
        <p><strong>Important:</strong> The information provided by this tool is not a substitute for professional medical advice. 
        If you're experiencing severe or persistent symptoms, please consult a healthcare professional immediately.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
