# Bio_ClinicalBERT Medical Advisor

This project implements a medical condition matcher using the Bio_ClinicalBERT model from Hugging Face, specifically trained on clinical text for improved medical domain performance.

## Overview

This implementation uses Bio_ClinicalBERT, a BERT model fine-tuned on clinical notes from MIMIC-III, to create contextual embeddings of medical symptoms and diagnoses. It then uses semantic similarity to match patient descriptions with potential medical conditions from a database.

## Features

- Uses **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) for semantic understanding of medical text
- Provides more contextual matching than TF-IDF approaches
- Can understand medical terminology and relationships better than general language models
- Includes both single-prediction and batch processing implementations
- GPU acceleration support for faster inference

## Installation

1. Install the required packages:
```bash
pip install -r requirements_bert.txt
```

2. Make sure your medical data CSV file is in the expected format with columns for Disease, Symptoms, Diagnosis, Treatment, and optionally Laymen Terms.

## Usage

### Basic Usage

```python
from bert_medical_model import BERTMedicalModel

# Initialize the model
model = BERTMedicalModel("combined_diseases_v2.csv")

# Make predictions
results = model.predict(
    symptoms="Persistent cough, fever, fatigue, shortness of breath",
    diagnosis="Chest X-ray shows infiltrates"
)

# Display results
for i, result in enumerate(results):
    print(f"{i+1}. {result['disease']} ({result['match_percentage']}% match)")
    print(f"   Treatment: {result['treatment'][:100]}...")
```

### Batch Processing for Multiple Queries

For processing multiple cases at once (faster than sequential processing):

```python
from batch_bert_medical_model import BatchBERTMedicalModel

# Initialize the model with batch size
model = BatchBERTMedicalModel("combined_diseases_v2.csv", batch_size=8)

# Prepare multiple cases
cases = [
    {"symptoms": "Persistent cough, fever", "diagnosis": "Chest X-ray shows infiltrates"},
    {"symptoms": "Headache, nausea, sensitivity to light", "diagnosis": ""},
    {"symptoms": "Frequent urination, increased thirst", "diagnosis": "Blood glucose 240 mg/dL"}
]

# Get predictions for all cases at once
batch_results = model.predict_batch(cases)

# Each element in batch_results corresponds to results for one case
```

## Benchmarking Against Other Models

To compare Bio_ClinicalBERT with TF-IDF models:

```bash
python bert_benchmark.py --data=combined_diseases_v2.csv --num-tests=20 --real-examples --models=all
```

Options:
- `--data`: Path to the medical data CSV file
- `--num-tests`: Number of random test cases to generate
- `--real-examples`: Include real-world examples in benchmark
- `--device`: Device to run BERT on (cuda, cpu, etc.)
- `--models`: Which models to benchmark (all, bert, basic, improved)
- `--output-dir`: Directory to save benchmark results

## Advantages of Bio_ClinicalBERT

1. **Domain-specific understanding**: Trained on clinical notes, it understands medical terminology better than general language models

2. **Contextual embeddings**: Captures the meaning of medical terms in context, rather than just keyword matching

3. **Semantic similarity**: Can identify conceptually related conditions even when symptoms are described differently

4. **Improved accuracy**: Typically provides more accurate matches than TF-IDF or other lexical matching approaches

## Performance Considerations

- The first run will download the Bio_ClinicalBERT model (about 700MB)
- Processing is significantly faster on GPU than CPU
- Batch processing can improve throughput when handling multiple queries
- The model requires more memory than TF-IDF approaches

## Extending the Model

To extend or customize this implementation:

1. **Fine-tune on your specific data**: You can further fine-tune Bio_ClinicalBERT on your specific medical corpus
2. **Add explainability**: Implement attention visualization to show which symptoms were most important for the match
3. **Improve preprocessing**: Add medical-specific text normalization (e.g., standardizing units, abbreviations)
4. **Add entity recognition**: Incorporate medical entity extraction to better structure symptom information

## References

- [Bio_ClinicalBERT on Hugging Face](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- Original paper: [ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission](https://arxiv.org/abs/1904.05342)
