import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Set page config
st.set_page_config(
    page_title="Medical Advisor",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for better visual presentation
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
    .match-high {
        background-color: #d4edda;
        border-color: #c3e6cb;
        border-left: 5px solid #28a745;
    }
    .match-medium {
        background-color: #fff3cd;
        border-color: #ffeeba;
        border-left: 5px solid #ffc107;
    }
    .match-low {
        background-color: #f8f9fa;
        border-color: #d6d8db;
        border-left: 5px solid #6c757d;
    }
    .disclaimer {
        font-size: 12px;
        color: #6c757d;
        font-style: italic;
    }
    .highlight {
        background-color: #ffeeba;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .symptom-tag {
        display: inline-block;
        background-color: #e2f0fd;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.85em;
    }
    .treatment-section {
        background-color: #f1f8e9;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text) or text == "":
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the medical data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/combined_diseases_v2.csv')
        # Preprocess text columns
        for col in ['Symptoms', 'Diagnosis', 'Disease', 'Treatment']:
            df[col] = df[col].apply(preprocess_text)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to extract key symptoms as tags
def extract_symptom_tags(symptom_text, max_tags=5):
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

# Function to highlight matching terms
def highlight_matches(text, query_terms, html=True):
    if not text or not query_terms:
        return text
    
    highlighted = text
    for term in query_terms:
        if len(term) > 3:  # Only highlight meaningful terms
            if html:
                highlighted = re.sub(f'(?i){re.escape(term)}', f'<span class="highlight">\\g<0></span>', highlighted)
            else:
                highlighted = re.sub(f'(?i){re.escape(term)}', f'**\\g<0>**', highlighted)
    
    return highlighted

# Improved function to find similar diseases with weighted search
@st.cache_data
def find_similar_diseases(symptoms, diagnosis, df, top_n=5, min_similarity=15):
    if not symptoms and not diagnosis:
        return []
    
    # Preprocess user input
    symptoms = preprocess_text(symptoms)
    diagnosis = preprocess_text(diagnosis)
    
    # Extract query terms for highlighting later
    query_terms = set()
    if symptoms:
        query_terms.update([term for term in symptoms.split() if len(term) > 3])
    if diagnosis:
        query_terms.update([term for term in diagnosis.split() if len(term) > 3])
    
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
                symptom_tags = extract_symptom_tags(df.iloc[idx]['Symptoms'])
                
                # Calculate match quality
                match_quality = "low"
                if similarity_percentage >= 50:
                    match_quality = "high"
                elif similarity_percentage >= 25:
                    match_quality = "medium"
                
                # Highlight matching terms
                highlighted_symptoms = highlight_matches(df.iloc[idx]['Symptoms'], query_terms, html=False)
                
                results.append({
                    'disease': df.iloc[idx]['Disease'],
                    'match_percentage': similarity_percentage,
                    'match_quality': match_quality,
                    'treatment': df.iloc[idx]['Treatment'],
                    'symptoms': df.iloc[idx]['Symptoms'],
                    'highlighted_symptoms': highlighted_symptoms,
                    'symptom_tags': symptom_tags,
                    'diagnosis': df.iloc[idx]['Diagnosis'],
                    'layman_terms': df.iloc[idx]['Laymen Terms']
                })
        
        return results
    except Exception as e:
        st.error(f"Error in search: {e}")
        return []

# Header with better styling
st.title("🩺 Medical Symptom Analyzer")
st.markdown("""
This tool helps identify possible medical conditions based on your symptoms and any diagnostic information you may have.
It searches a database of medical conditions to find potential matches.
""")

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
                   help="Example: Persistent cough, fever, fatigue, shortness of breath",
                   height=150)

diagnosis_info = st.sidebar.text_area("Any diagnostic information (if available):", 
                         help="Example: X-ray showed lung infiltrates, elevated white blood cell count",
                         height=100)

severity = st.sidebar.select_slider("Rate the severity of your symptoms:", 
                          options=["Very mild", "Mild", "Moderate", "Severe", "Very severe"],
                          value="Moderate")

duration = st.sidebar.selectbox("How long have you been experiencing these symptoms?", 
                       ["Less than a day", "1-3 days", "4-7 days", "1-2 weeks", 
                        "2-4 weeks", "1-3 months", "More than 3 months"])

min_similarity = st.sidebar.slider("Minimum match quality (%):", 10, 50, 15, 
                         help="Higher values show fewer but more relevant results")

analyze_button = st.sidebar.button("Analyze Symptoms", type="primary", use_container_width=True)

# Main content area
if analyze_button:
    if not symptoms and not diagnosis_info:
        st.warning("Please describe your symptoms or provide diagnostic information.")
    else:
        with st.spinner("Analyzing your information..."):
            # Load medical data
            medical_data = load_data()
            
            if medical_data.empty:
                st.error("Could not load the medical database. Please check if the CSV file exists.")
            else:
                # Find similar diseases
                results = find_similar_diseases(symptoms, diagnosis_info, medical_data, 
                                                top_n=10, min_similarity=min_similarity)
                
                if results:
                    st.success(f"Found {len(results)} potential matches for your symptoms")
                    
                    # Display search summary
                    symptom_text = symptoms if symptoms else "Not provided"
                    diagnosis_text = diagnosis_info if diagnosis_info else "Not provided"
                    
                    with st.expander("Your search details", expanded=False):
                        st.write(f"**Symptoms:** {symptom_text}")
                        st.write(f"**Diagnostic information:** {diagnosis_text}")
                        st.write(f"**Severity:** {severity}")
                        st.write(f"**Duration:** {duration}")
                    
                    # Display results in a visually appealing way
                    for result in results:
                        match_class = f"result-box match-{result['match_quality']}"
                        
                        st.markdown(f"""
                        <div class="{match_class}">
                            <h3>{result['disease'].title()}</h3>
                            <p><strong>Match confidence:</strong> {result['match_percentage']}%</p>
                        """, unsafe_allow_html=True)
                        
                        # Display symptom tags
                        if result['symptom_tags']:
                            st.markdown("<p><strong>Key symptoms:</strong></p>", unsafe_allow_html=True)
                            tags_html = " ".join([f'<span class="symptom-tag">{tag}</span>' for tag in result['symptom_tags']])
                            st.markdown(tags_html, unsafe_allow_html=True)
                        
                        # Show treatment in a highlighted section
                        st.markdown(f"""
                            <div class="treatment-section">
                                <p><strong>Typical treatment:</strong> {result['treatment']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # More details in an expander
                        with st.expander("View medical details"):
                            st.markdown(f"**Symptoms:** {result['highlighted_symptoms']}")
                            st.write("**Diagnosis method:**", result['diagnosis'])
                            st.write("**Treatment:**", result['treatment'])
                            if 'layman_terms' in result and result['layman_terms']:
                                st.write("**In simple terms:**", result['layman_terms'])
                else:
                    st.warning("""
                    No matching conditions found based on the provided information. 
                    Try providing more detailed symptoms or lowering the minimum match quality.
                    """)
else:
    st.info("Enter your symptoms and click 'Analyze Symptoms' to get started.")
    
    # Show some example queries
    with st.expander("Example searches"):
        st.markdown("""
        **Example 1: Common Cold**
        - Symptoms: Runny nose, sore throat, coughing, mild fever, fatigue
        - Diagnosis: None
        
        **Example 2: Potential Diabetes**
        - Symptoms: Frequent urination, increased thirst, unexplained weight loss, fatigue
        - Diagnosis: Elevated blood sugar levels
        
        **Example 3: Migraine**
        - Symptoms: Severe headache on one side, sensitivity to light and sound, nausea
        - Diagnosis: None
        """)

# Reminder about seeking professional advice
st.markdown("""
<div class="disclaimer">
    <p><strong>Important:</strong> The information provided by this tool is not a substitute for professional medical advice. 
    If you're experiencing severe or persistent symptoms, please consult a healthcare professional immediately.</p>
</div>
""", unsafe_allow_html=True)