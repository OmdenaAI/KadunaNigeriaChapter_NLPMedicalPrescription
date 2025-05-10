import streamlit as st
import pandas as pd
import openai
import os
import time
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page config
st.set_page_config(
    page_title="Agentic Medical Advisor",
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
    .step-container {
        margin-top: 20px;
        padding: 10px;
        border-left: 3px solid #4CAF50;
        background-color: #f9f9f9;
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

# Define the Medical Agent
class MedicalAgent:
    def __init__(self, medical_data):
        self.medical_data = medical_data
        self.vectorizer = None
        self.tfidf_matrix = None
        self.history = []
        
    def _preprocess_data(self):
        """Preprocess the medical data for vector search"""
        # Combine symptoms and diagnosis for comparison
        self.medical_data['Combined'] = self.medical_data['Symptoms'] + ' ' + self.medical_data['Diagnosis']
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.medical_data['Combined'].tolist())
        
    def search_database(self, query, top_n=5):
        """Search the database using vector similarity"""
        if self.vectorizer is None:
            self._preprocess_data()
            
        # Create a vector for the user query
        query_vector = self.vectorizer.transform([query])
        
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
                    'disease': self.medical_data.iloc[idx]['Disease'],
                    'match_percentage': similarity_percentage,
                    'treatment': self.medical_data.iloc[idx]['Treatment'],
                    'symptoms': self.medical_data.iloc[idx]['Symptoms'],
                    'diagnosis': self.medical_data.iloc[idx]['Diagnosis'],
                    'layman_terms': self.medical_data.iloc[idx]['Laymen Terms']
                })
        
        return results
    
    def analyze_with_llm(self, symptoms, diagnosis, database_results):
        """Use LLM to analyze symptoms and database results"""
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
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",  # or other appropriate model
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
                st.error("Failed to parse AI response. Using database results only.")
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
            st.error(f"Error querying AI: {str(e)}")
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
    
    def get_condition_details(self, condition_name):
        """Get details for a specific condition from the database"""
        matches = self.medical_data[self.medical_data['Disease'].str.contains(condition_name, case=False, na=False)]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None
    
    def process_case(self, symptoms, diagnosis, show_thinking=False):
        """Process a medical case with both vector search and LLM analysis"""
        # Log the inputs
        self.history.append({
            "symptoms": symptoms,
            "diagnosis": diagnosis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Step 1: Search the database
        if show_thinking:
            with st.expander("Step 1: Database Search", expanded=True):
                st.write("Searching medical database for matches...")
                
        combined_query = f"{symptoms} {diagnosis}"
        database_results = self.search_database(combined_query)
        
        if show_thinking:
            st.write(f"Found {len(database_results)} potential matches.")
            
        # Step 2: Analyze with LLM
        if show_thinking:
            with st.expander("Step 2: AI Analysis", expanded=True):
                st.write("Analyzing symptoms and database results with AI...")
                
        llm_analysis = self.analyze_with_llm(symptoms, diagnosis, database_results)
        
        # Step 3: Combine results
        if show_thinking:
            with st.expander("Step 3: Final Results", expanded=True):
                st.write("Combining database matches with AI analysis...")
        
        # Prepare the final results
        final_results = {
            "database_matches": database_results,
            "llm_analysis": llm_analysis
        }
        
        return final_results

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
    
if 'results' not in st.session_state:
    st.session_state.results = None
    
if 'show_thinking' not in st.session_state:
    st.session_state.show_thinking = False

# Header
st.title("🩺 Agentic Medical Advisor")
st.markdown("This application uses AI to find potential matches for symptoms and diagnostic information from a medical database.")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <p><strong>Disclaimer:</strong> This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Input form
st.sidebar.title("Patient Information")

symptoms = st.sidebar.text_area("Describe your symptoms in detail:", 
                       help="Example: Persistent cough, fever, fatigue, and shortness of breath")

diagnosis_info = st.sidebar.text_area("Any diagnostic information (if available):", 
                             help="Example: X-ray showed lung infiltrates, elevated white blood cell count")

st.sidebar.markdown("---")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    st.session_state.show_thinking = st.checkbox("Show AI thinking process", value=False)
    use_openai = st.checkbox("Use OpenAI for enhanced analysis", value=True)
    
    if use_openai:
        api_key = st.text_input("OpenAI API Key (leave empty to use env variable)", type="password")
        if api_key:
            openai.api_key = api_key

# Process button
if st.sidebar.button("Analyze Symptoms"):
    if not symptoms:
        st.warning("Please describe your symptoms.")
    else:
        # Load data and initialize agent if needed
        if st.session_state.agent is None:
            with st.spinner("Loading medical database..."):
                medical_data = load_data()
                st.session_state.agent = MedicalAgent(medical_data)
        
        # Process the case
        with st.spinner("Analyzing your information..."):
            st.session_state.results = st.session_state.agent.process_case(
                symptoms, 
                diagnosis_info,
                show_thinking=st.session_state.show_thinking
            )

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    # Display likely conditions
    st.header("Potential Conditions")
    
    likely_conditions = results["llm_analysis"]["likely_conditions"]
    
    if not likely_conditions:
        st.warning("No conditions could be determined from the provided information.")
    
    for condition in likely_conditions:
        with st.container():
            st.markdown(f"""
            <div class="result-box">
                <h3>{condition['name']}</h3>
                <p><strong>Confidence:</strong> {condition['confidence_percentage']}%</p>
                <p><strong>Analysis:</strong> {condition['reasoning']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Find database details for this condition
            found_match = False
            for db_match in results["database_matches"]:
                if condition['name'].lower() in db_match['disease'].lower():
                    found_match = True
                    with st.expander(f"Medical details for {db_match['disease']}"):
                        st.write("**Symptoms:**", db_match['symptoms'])
                        st.write("**Diagnosis method:**", db_match['diagnosis'])
                        st.write("**Treatment:**", db_match['treatment'])
                        st.write("**In simple terms:**", db_match['layman_terms'])
            
            if not found_match:
                st.info(f"No detailed information available in database for {condition['name']}.")
    
    # Additional considerations
    additional = results["llm_analysis"].get("additional_considerations", [])
    if additional:
        st.header("Additional Considerations")
        st.write("The following conditions might also be worth investigating based on the symptoms:")
        
        for condition in additional:
            st.markdown(f"- **{condition}**")
    
    # Patient advice
    st.header("General Advice")
    st.markdown(f"_{results['llm_analysis'].get('patient_advice', 'Please consult a healthcare professional for proper diagnosis and treatment.')}_")
    
    # Disclaimer reminder
    st.markdown("""
    <div class="disclaimer">
        <p><strong>Important:</strong> The information provided by this tool is not a substitute for professional medical advice. 
        If you're experiencing severe or persistent symptoms, please consult a healthcare professional immediately.</p>
    </div>
    """, unsafe_allow_html=True)
