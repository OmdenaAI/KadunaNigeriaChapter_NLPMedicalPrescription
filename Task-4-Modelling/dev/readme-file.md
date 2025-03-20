# Medical Advisor LLM App

This repository contains three versions of a medical advisor application built with Python, Streamlit, and AI capabilities to match symptoms and diagnoses with potential diseases and treatments.

## Warning & Disclaimer

**This application is for educational and informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Features

- Input symptoms and diagnostic information
- Get potential disease matches from a medical database
- Receive treatment suggestions based on matched diseases
- Advanced agentic version combines database searching with LLM reasoning

## Project Versions

### 1. Full Medical Advisor App
- Uses OpenAI's API for enhanced analysis
- Requires an API key
- Provides detailed analysis and matching

### 2. Simplified Medical Advisor
- No API key required
- Uses basic TF-IDF vectorization and cosine similarity for matching
- Suitable for quick testing

### 3. Agentic Medical Advisor
- Combines local database search with LLM reasoning
- Shows the AI's thinking process (optional)
- Provides detailed analysis and additional considerations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medical-advisor-app.git
cd medical-advisor-app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. (Optional) Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Simplified App (No API Key Required)
```bash
streamlit run simplified_medical_app.py
```

### Running the Full App (Requires OpenAI API Key)
```bash
streamlit run medical_advisor_app.py
```

### Running the Agentic App
```bash
streamlit run agentic_medical_app.py
```

## Data

The application uses a CSV database of diseases with the following columns:
- Disease: Name of the disease
- Treatment: Recommended treatments
- Diagnosis: How the disease is diagnosed
- Symptoms: Common symptoms
- Laymen Terms: Simplified explanation

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit==1.31.1
pandas==2.2.0
scikit-learn==1.4.0
openai==1.12.0
python-dotenv==1.0.1
```

## Project Structure

```
medical-advisor-app/
│
├── data/
│   └── combined_diseases_v2.csv          # Disease database
│
├── medical_advisor_app.py                # Full app
├── simplified_medical_app.py             # Simplified app (no API)
├── agentic_medical_app.py                # Advanced agentic app
│
├── .env                                  # Environment variables (not in repo)
├── requirements.txt                      # Package dependencies
└── README.md                             # This file
```

## Extending the Project

### Adding More Diseases
You can extend the CSV database with more diseases and their information. Ensure the format matches the existing structure.

### Improving the Agent
The MedicalAgent class in the agentic app can be extended with:
- Better similarity algorithms
- More sophisticated reasoning
- Multi-step dialog capabilities
- History tracking and follow-up suggestions

### UI Enhancements
Consider adding:
- Interactive visualizations for symptom matching
- Better search functionality
- Patient history management
- PDF report generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
