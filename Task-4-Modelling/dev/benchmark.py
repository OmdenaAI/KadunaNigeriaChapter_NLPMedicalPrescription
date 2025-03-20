#!/usr/bin/env python3
"""
Benchmark script to compare different medical advisor models
Updated to use the new OpenAI API syntax
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import argparse
import os
import json
from datetime import datetime

# Import the models
from basic_model import BasicMedicalModel
from improved_model import ImprovedMedicalModel

# Import from the updated agentic model file
from agentic_model import AgenticMedicalModel


# Setup logging
class TeeLogger:
    """Class to log output to both console and a file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark medical advisor models")
    parser.add_argument("--data", type=str, default="../data/combined_diseases_v2.csv",
                        help="Path to the medical data CSV file")
    parser.add_argument("--num-tests", type=int, default=50,
                        help="Number of random test cases to generate")
    parser.add_argument("--real-examples", action="store_true",
                        help="Include real-world examples in benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Use mock LLM instead of OpenAI API for agentic model")
    return parser.parse_args()

def generate_test_cases(df, num_cases=20):
    """
    Generate random test cases from the dataset
    
    Args:
        df (pandas.DataFrame): DataFrame with medical data
        num_cases (int): Number of test cases to generate
        
    Returns:
        list: List of test case dictionaries
    """
    test_cases = []
    
    for _ in range(num_cases):
        # Randomly select a disease
        idx = np.random.randint(0, len(df))
        disease = df.iloc[idx]['Disease']
        symptoms = df.iloc[idx]['Symptoms']
        diagnosis = df.iloc[idx]['Diagnosis']
        
        # Extract a random subset of symptoms (50-90%)
        symptom_parts = symptoms.split(',')
        num_parts = max(1, int(len(symptom_parts) * np.random.uniform(0.5, 0.9)))
        selected_symptoms = ','.join(np.random.choice(symptom_parts, num_parts, replace=False))
        
        # Only use diagnosis information in 50% of cases
        if np.random.random() < 0.5:
            selected_diagnosis = ""
        else:
            diagnosis_parts = diagnosis.split(',')
            num_parts = max(1, int(len(diagnosis_parts) * np.random.uniform(0.3, 0.8)))
            selected_diagnosis = ','.join(np.random.choice(diagnosis_parts, num_parts, replace=False))
        
        test_cases.append({
            'true_disease': disease,
            'symptoms': selected_symptoms,
            'diagnosis': selected_diagnosis
        })
    
    return test_cases

def get_real_test_cases():
    """
    Provide a set of real-world test cases
    
    Returns:
        list: List of real-world test cases
    """
    return [
        {
            "name": "Common Cold",
            "true_disease": "Cold",
            "symptoms": "Runny nose, sore throat, coughing, mild fever, fatigue",
            "diagnosis": ""
        },
        {
            "name": "Diabetes Type 2",
            "true_disease": "Type 2 Diabetes",
            "symptoms": "Frequent urination, increased thirst, unexplained weight loss, fatigue, blurred vision",
            "diagnosis": "Blood glucose level 240 mg/dL"
        },
        {
            "name": "Migraine",
            "true_disease": "Migraine",
            "symptoms": "Severe headache on one side, sensitivity to light and sound, nausea, vision changes",
            "diagnosis": ""
        },
        {
            "name": "Pneumonia",
            "true_disease": "Pneumonia",
            "symptoms": "High fever, cough with phlegm, difficulty breathing, chest pain, fatigue",
            "diagnosis": "Chest X-ray shows infiltrates in lower left lobe"
        },
        {
            "name": "Appendicitis", 
            "true_disease": "Appendicitis",
            "symptoms": "Sharp pain in lower right abdomen, nausea, vomiting, loss of appetite, low fever",
            "diagnosis": "White blood cell count elevated"
        }
    ]

def evaluate_result(results, true_disease):
    """
    Evaluate if the model found the correct disease
    
    Args:
        results: Model prediction results
        true_disease (str): The correct disease name
        
    Returns:
        dict: Evaluation metrics
    """
    # Handle different result formats
    if isinstance(results, dict) and "conditions" in results:
        # Agentic model format
        conditions = results["conditions"]
    elif isinstance(results, list):
        # Basic and improved model format
        conditions = results
    else:
        return {"found": False, "rank": None, "confidence": 0}
    
    # Check if the true disease is in the results
    found = False
    rank = None
    confidence = 0
    
    for i, res in enumerate(conditions):
        if true_disease.lower() in res['disease'].lower():
            found = True
            rank = i + 1
            confidence = res['match_percentage']
            break
    
    return {
        'found': found,
        'rank': rank,
        'confidence': confidence
    }

def visualize_results(results, output_dir):
    """
    Create visualizations of benchmark results
    
    Args:
        results (dict): Benchmark results dictionary
        output_dir (str): Directory to save visualizations
    """
    models = list(results.keys())
    
    # Accuracy comparison
    accuracies = [results[model]['accuracy'] * 100 for model in models]
    
    # Average rank comparison (lower is better)
    avg_ranks = [results[model]['avg_rank'] if results[model]['avg_rank'] is not None else 0 for model in models]
    
    # Average confidence comparison
    avg_confidences = [results[model]['avg_confidence'] if results[model]['avg_confidence'] is not None else 0 for model in models]
    
    # Average processing time comparison
    avg_times = [results[model]['avg_time'] * 1000 for model in models]  # Convert to ms
    
    # Create the visualization
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy plot
    axs[0, 0].bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0, 0].set_title('Accuracy (%)')
    axs[0, 0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axs[0, 0].text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Average rank plot
    axs[0, 1].bar(models, avg_ranks, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0, 1].set_title('Average Rank (lower is better)')
    for i, v in enumerate(avg_ranks):
        axs[0, 1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Average confidence plot
    axs[1, 0].bar(models, avg_confidences, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[1, 0].set_title('Average Confidence (%)')
    for i, v in enumerate(avg_confidences):
        axs[1, 0].text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Average time plot
    axs[1, 1].bar(models, avg_times, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[1, 1].set_title('Average Processing Time (ms)')
    for i, v in enumerate(avg_times):
        axs[1, 1].text(i, v + 1, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_summary.png'))
    plt.close()
    
    # Print results summary to console
    print("\nBenchmark Results:")
    print("=================")
    
    for model in models:
        print(f"\nModel: {model}")
        print(f"  Accuracy: {results[model]['accuracy'] * 100:.1f}%")
        print(f"  Average Rank: {results[model]['avg_rank'] if results[model]['avg_rank'] is not None else 'N/A'}")
        print(f"  Average Confidence: {results[model]['avg_confidence'] if results[model]['avg_confidence'] is not None else 'N/A':.1f}%")
        print(f"  Average Processing Time: {results[model]['avg_time'] * 1000:.1f} ms")

def visualize_real_results(results, output_dir):
    """
    Create visualizations of real-world test results
    
    Args:
        results (dict): Benchmark results dictionary
        output_dir (str): Directory to save visualizations
    """
    models = list(results.keys())
    
    # Extract real test case results
    real_results = {}
    for model in models:
        real_results[model] = results[model]['real_results']
    
    # Get case names
    case_names = [r['case'] for r in real_results[models[0]]]
    
    # Prepare data
    success_data = np.zeros((len(models), len(case_names)))
    time_data = np.zeros((len(models), len(case_names)))
    
    for i, model in enumerate(models):
        for j, case in enumerate(real_results[model]):
            success_data[i, j] = 1 if case['found'] else 0
            time_data[i, j] = case['time'] * 1000  # Convert to ms
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Success plot (1 for success, 0 for failure)
    im1 = ax1.imshow(success_data, cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(case_names)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(case_names)
    ax1.set_yticklabels(models)
    ax1.set_title('Success by Case (Green = Found, Red = Not Found)')
    
    # Add text annotations to the success plot
    for i in range(len(models)):
        for j in range(len(case_names)):
            text = "✓" if success_data[i, j] == 1 else "✗"
            ax1.text(j, i, text, ha="center", va="center", color="black")
    
    # Time plot
    im2 = ax2.imshow(time_data, cmap='Blues')
    ax2.set_xticks(np.arange(len(case_names)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(case_names)
    ax2.set_yticklabels(models)
    ax2.set_title('Processing Time by Case (ms)')
    
    # Add text annotations to the time plot
    for i in range(len(models)):
        for j in range(len(case_names)):
            ax2.text(j, i, f"{time_data[i, j]:.1f}", ha="center", va="center", color="white")
    
    # Add colorbar
    cbar = ax2.figure.colorbar(im2, ax=ax2)
    cbar.ax.set_ylabel("Time (ms)", rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_examples_comparison.png'))
    plt.close()
    
    # Calculate summary statistics
    summary = {}
    for model in models:
        success_rate = sum(1 for r in real_results[model] if r['found']) / len(real_results[model]) * 100
        avg_time = sum(r['time'] * 1000 for r in real_results[model]) / len(real_results[model])
        summary[model] = {
            "success_rate": success_rate,
            "avg_time": avg_time
        }
    
    # Create summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate chart
    success_rates = [summary[model]['success_rate'] for model in models]
    ax1.bar(models, success_rates, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax1.set_ylim(0, 100)
    ax1.set_title('Real Examples Success Rate (%)')
    ax1.set_ylabel('Percentage')
    
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # Time chart
    times = [summary[model]['avg_time'] for model in models]
    ax2.bar(models, times, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax2.set_title('Real Examples Average Processing Time (ms)')
    ax2.set_ylabel('Time (ms)')
    
    for i, v in enumerate(times):
        ax2.text(i, v + 1, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_examples_summary.png'))
    plt.close()
    
    # Print real examples summary
    print("\nReal-World Examples Results:")
    print("===========================")
    
    for model in models:
        print(f"\nModel: {model}")
        print(f"  Success Rate: {summary[model]['success_rate']:.1f}%")
        print(f"  Average Processing Time: {summary[model]['avg_time']:.1f} ms")
        
        # List successful and failed cases
        successful = [r['case'] for r in real_results[model] if r['found']]
        failed = [r['case'] for r in real_results[model] if not r['found']]
        
        if successful:
            print(f"  Successful cases: {', '.join(successful)}")
        if failed:
            print(f"  Failed cases: {', '.join(failed)}")

def run_benchmark(data_path, num_test_cases=50, include_real=True, use_mock_llm=True, output_dir="benchmark_results"):
    """
    Run the benchmarking process on all models
    
    Args:
        data_path (str): Path to the medical data CSV
        num_test_cases (int): Number of random test cases to generate
        include_real (bool): Whether to include real-world examples
        use_mock_llm (bool): Whether to use mock LLM for the agentic model
        output_dir (str): Directory to save results
        
    Returns:
        tuple: Results dictionary and visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    
    if df.empty:
        print("Failed to load data!")
        return
    
    # Initialize models
    print("Initializing models...")
    basic_model = BasicMedicalModel(data_path)
    improved_model = ImprovedMedicalModel(data_path)
    agentic_model = AgenticMedicalModel(data_path, use_mock_llm=use_mock_llm)
    
    # Generate test cases
    print(f"Generating {num_test_cases} random test cases...")
    random_test_cases = generate_test_cases(df, num_test_cases)
    
    # Save test cases for inspection
    test_df = pd.DataFrame(random_test_cases)
    test_df.to_csv(os.path.join(output_dir, 'random_test_cases.csv'), index=False)
    print(f"Random test cases saved to {os.path.join(output_dir, 'random_test_cases.csv')}")
    
    # Get real test cases if requested
    real_test_cases = []
    if include_real:
        real_test_cases = get_real_test_cases()
        with open(os.path.join(output_dir, 'real_test_cases.json'), 'w') as f:
            json.dump(real_test_cases, f, indent=2)
        print(f"Real test cases saved to {os.path.join(output_dir, 'real_test_cases.json')}")
    
    # Initialize results containers
    results = {
        'Basic Model': {'found': 0, 'avg_rank': [], 'avg_confidence': [], 'times': [], 'real_results': []},
        'Improved Model': {'found': 0, 'avg_rank': [], 'avg_confidence': [], 'times': [], 'real_results': []},
        'Agentic Model': {'found': 0, 'avg_rank': [], 'avg_confidence': [], 'times': [], 'real_results': []}
    }
    
    # Run benchmarks on random test cases
    print("Running benchmarks on random test cases...")
    for i, case in enumerate(tqdm(random_test_cases)):
        true_disease = case['true_disease']
        symptoms = case['symptoms']
        diagnosis = case['diagnosis']
        
        # Test basic model
        start_time = time.time()
        result1 = basic_model.predict(symptoms, diagnosis)
        time1 = time.time() - start_time
        eval1 = evaluate_result(result1, true_disease)
        
        # Test improved model
        start_time = time.time()
        result2 = improved_model.predict(symptoms, diagnosis)
        time2 = time.time() - start_time
        eval2 = evaluate_result(result2, true_disease)
        
        # Test agentic model
        start_time = time.time()
        result3 = agentic_model.predict(symptoms, diagnosis)
        time3 = time.time() - start_time
        eval3 = evaluate_result(result3, true_disease)
        
        # Record results
        if eval1['found']:
            results['Basic Model']['found'] += 1
            results['Basic Model']['avg_rank'].append(eval1['rank'])
            results['Basic Model']['avg_confidence'].append(eval1['confidence'])
        results['Basic Model']['times'].append(time1)
        
        if eval2['found']:
            results['Improved Model']['found'] += 1
            results['Improved Model']['avg_rank'].append(eval2['rank'])
            results['Improved Model']['avg_confidence'].append(eval2['confidence'])
        results['Improved Model']['times'].append(time2)
        
        if eval3['found']:
            results['Agentic Model']['found'] += 1
            results['Agentic Model']['avg_rank'].append(eval3['rank'])
            results['Agentic Model']['avg_confidence'].append(eval3['confidence'])
        results['Agentic Model']['times'].append(time3)
    
    # Run benchmarks on real test cases
    if include_real:
        print("Running benchmarks on real-world examples...")
        for case in real_test_cases:
            true_disease = case['true_disease']
            symptoms = case['symptoms']
            diagnosis = case['diagnosis']
            
            # Test basic model
            start_time = time.time()
            result1 = basic_model.predict(symptoms, diagnosis)
            time1 = time.time() - start_time
            eval1 = evaluate_result(result1, true_disease)
            
            # Test improved model
            start_time = time.time()
            result2 = improved_model.predict(symptoms, diagnosis)
            time2 = time.time() - start_time
            eval2 = evaluate_result(result2, true_disease)
            
            # Test agentic model
            start_time = time.time()
            result3 = agentic_model.predict(symptoms, diagnosis)
            time3 = time.time() - start_time
            eval3 = evaluate_result(result3, true_disease)
            
            # Record real case results
            results['Basic Model']['real_results'].append({
                'case': case['name'],
                'found': eval1['found'],
                'rank': eval1['rank'],
                'confidence': eval1['confidence'],
                'time': time1
            })
            
            results['Improved Model']['real_results'].append({
                'case': case['name'],
                'found': eval2['found'],
                'rank': eval2['rank'],
                'confidence': eval2['confidence'],
                'time': time2
            })
            
            results['Agentic Model']['real_results'].append({
                'case': case['name'],
                'found': eval3['found'],
                'rank': eval3['rank'],
                'confidence': eval3['confidence'],
                'time': time3
            })
    
    # Calculate final metrics for random test cases
    for model in results:
        results[model]['accuracy'] = results[model]['found'] / num_test_cases
        results[model]['avg_rank'] = np.mean(results[model]['avg_rank']) if results[model]['avg_rank'] else None
        results[model]['avg_confidence'] = np.mean(results[model]['avg_confidence']) if results[model]['avg_confidence'] else None
        results[model]['avg_time'] = np.mean(results[model]['times'])
    
    # Save detailed results as JSON
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    visualize_results(results, output_dir)
    
    # Create results dataframe for CSV export
    result_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy (%)': [results[model]['accuracy'] * 100 for model in results],
        'Average Rank': [results[model]['avg_rank'] if results[model]['avg_rank'] is not None else 0 for model in results],
        'Average Confidence (%)': [results[model]['avg_confidence'] if results[model]['avg_confidence'] is not None else 0 for model in results],
        'Average Time (ms)': [results[model]['avg_time'] * 1000 for model in results]
    })
    
    result_df.to_csv(os.path.join(output_dir, 'benchmark_summary.csv'), index=False)
    print(f"Benchmark summary saved to {os.path.join(output_dir, 'benchmark_summary.csv')}")
    
    # If real examples were included, create a separate visualization
    if include_real:
        visualize_real_results(results, output_dir)
    
    return results

def main():
    """Main entry point for the benchmark script"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    
    # Set up logging
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "benchmark_log.txt")
    sys.stdout = TeeLogger(log_file)
    
    # Print start message
    print("=" * 60)
    print(f"Medical Advisor Models Benchmark - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"Data file: {args.data}")
    print(f"Number of random test cases: {args.num_tests}")
    print(f"Include real-world examples: {args.real_examples}")
    print(f"Use mock LLM: {args.mock_llm}")
    print(f"Results will be saved to: {output_dir}")
    print("=" * 60)
    
    # Run the benchmark
    results = run_benchmark(
        data_path=args.data,
        num_test_cases=args.num_tests,
        include_real=args.real_examples,
        use_mock_llm=args.mock_llm,
        output_dir=output_dir
    )
    
    # Print completion message
    print("\nBenchmark completed.")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()