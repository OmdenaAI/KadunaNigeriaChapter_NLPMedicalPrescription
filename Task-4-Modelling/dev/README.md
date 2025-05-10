# Medical Advisor Models

This repository contains three different implementations of medical advisor models that can match symptoms and diagnosis information with potential diseases and treatments.

## Models

### 1. Basic Model
A simple model using TF-IDF vectorization and cosine similarity to match symptoms with diseases.

### 2. Improved Model
An enhanced model with improved text preprocessing, symptom weighting, and bigram analysis for better matching.

### 3. Agentic Model
An advanced model that combines database searching with LLM reasoning for more contextual and accurate matches.

## Files

- `basic_model.py` - Implementation of the Basic Model
- `improved_model.py` - Implementation of the Improved Model
- `agentic_model.py` - Implementation of the Agentic Model
- `benchmark.py` - Script to benchmark and compare all three models

## Installation

1. Install the required packages:
```
pip install pandas numpy scikit-learn matplotlib tqdm python-dotenv openai
```

2. Place your medical data CSV file in the repository directory (default expected filename: `combined_diseases_v2.csv`).

3. (Optional) For the Agentic Model with OpenAI API, set your API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running Individual Models

Each model can be run independently for testing:

```python
# Basic Model
python basic_model.py

# Improved Model
python improved_model.py

# Agentic Model
python agentic_model.py
```

### Using Models in Your Code

```python
# Basic Model
from basic_model import BasicMedicalModel

model = BasicMedicalModel("combined_diseases_v2.csv")
results = model.predict("fever, cough, fatigue", "chest x-ray shows infiltrates")
print(results)

# Improved Model
from improved_model import ImprovedMedicalModel

model = ImprovedMedicalModel("combined_diseases_v2.csv")
results = model.predict("fever, cough, fatigue", "chest x-ray shows infiltrates")
print(results)

# Agentic Model
from agentic_model import AgenticMedicalModel

# With mock LLM (no API key needed)
model = AgenticMedicalModel("combined_diseases_v2.csv", use_mock_llm=True)
# Or with real OpenAI API
# model = AgenticMedicalModel("combined_diseases_v2.csv", use_mock_llm=False)

results = model.predict("fever, cough, fatigue", "chest x-ray shows infiltrates")
print(results)
```

### Running Benchmarks

To benchmark all three models and compare their performance:

```
python benchmark.py
```

Optional arguments:
- `--data` - Path to the medical data CSV file (default: "combined_diseases_v2.csv")
- `--num-tests` - Number of random test cases to generate (default: 50)
- `--real-examples` - Include real-world examples in benchmark
- `--output-dir` - Directory to save benchmark results (default: "benchmark_results")
- `--mock-llm` - Use mock LLM instead of OpenAI API for agentic model

Example:
```
python benchmark.py --data=my_dataset.csv --num-tests=100 --real-examples --mock-llm
```

## Benchmark Results

The benchmark script will generate several files in the output directory:

- `benchmark_log.txt` - Log of the benchmark run
- `benchmark_summary.csv` - CSV file with overall performance metrics
- `benchmark_results.json` - Detailed JSON results for all tests
- `benchmark_summary.png` - Visual comparison of model performance
- `real_examples_comparison.png` - Detailed comparison on real-world examples
- `real_examples_summary.png` - Summary of real-world example performance

## Expected Output

The benchmark results will include:

1. **Accuracy** - Percentage of cases where the model correctly identified the disease
2. **Average Rank** - Average position of the correct disease in the results (lower is better)
3. **Average Confidence** - Average confidence score for the correct disease
4. **Processing Time** - Average time to process each query in milliseconds

## Data Format

The expected format for the medical data CSV file is:

| Disease | Treatment | Diagnosis | Symptoms | Laymen Terms |
|---------|-----------|-----------|----------|--------------|
| Disease name | Treatment details | Diagnosis methods | Symptom list | Simplified explanation |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only and is not intended to be used for medical diagnosis or treatment. Always consult with a qualified healthcare provider for medical advice.
