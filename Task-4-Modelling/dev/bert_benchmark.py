#!/usr/bin/env python3
"""
Benchmark script to compare BERT model with TF-IDF models
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
try:
    from basic_model import BasicMedicalModel
    from improved_model import ImprovedMedicalModel
    from bert_medical_model import BERTMedicalModel
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Make sure all model files are in the current directory or in PYTHONPATH")
    sys.exit(1)

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
        
    def close(self):
        self.log.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark BERT model against TF-IDF models")
    parser.add_argument("--data", type=str, default="combined_diseases_v2.csv",
                        help="Path to the medical data CSV file")
    parser.add_argument("--num-tests", type=int, default=20,
                        help="Number of random test cases to generate")
    parser.add_argument("--real-examples", action="store_true",
                        help="Include real-world examples in benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run BERT on (cuda, cpu, etc.)")
    parser.add_argument("--models", type=str, default="all",
                        help="Which models to benchmark (all, bert, basic, improved)")
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
        # Basic, improved, and BERT model format
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
    bars1 = axs[0, 0].bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0, 0].set_title('Accuracy (%)')
    axs[0, 0].set_ylim(0, 100)
    for bar in bars1:
        height = bar.get_height()
        axs[0, 0].text(bar.get_x() + bar.get_width()/2., height + 2,
                f"{height:.1f}%", ha='center', va='bottom')
    
    # Average rank plot
    bars2 = axs[0, 1].bar(models, avg_ranks, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[0, 1].set_title('Average Rank (lower is better)')
    for bar in bars2:
        height = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.2f}", ha='center', va='bottom')
    
    # Average confidence plot
    bars3 = axs[1, 0].bar(models, avg_confidences, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[1, 0].set_title('Average Confidence (%)')
    for bar in bars3:
        height = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width()/2., height + 2,
                f"{height:.1f}%", ha='center', va='bottom')
    
    # Average time plot
    bars4 = axs[1, 1].bar(models, avg_times, color=['#3498db', '#2ecc71', '#e74c3c'])
    axs[1, 1].set_title('Average Processing Time (ms)')
    axs[1, 1].set_yscale('log')  # Use log scale for time
    for bar in bars4:
        height = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f"{height:.1f}", ha='center', va='bottom')
    
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
    ax2.set_yscale('log')  # Use log scale for time
    
    for i, v in enumerate(times):
        ax2.text(i, v * 1.1, f"{v:.1f}", ha='center')
    
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

def run_benchmark(data_path, num_test_cases=20, include_real=True, device=None, models_to_run="all", output_dir="benchmark_results"):
    """
    Run the benchmarking process comparing models
    
    Args:
        data_path (str): Path to the medical data CSV
        num_test_cases (int): Number of random test cases to generate
        include_real (bool): Whether to include real-world examples
        device (str): Device to run BERT on (cuda, cpu, etc.)
        models_to_run (str): Which models to benchmark (all, bert, basic, improved)
        output_dir (str): Directory to save results
        
    Returns:
        dict: Results dictionary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    
    if df.empty:
        print("Failed to load data!")
        return
    
    # Initialize models based on user selection
    model_dict = {}
    
    if models_to_run.lower() in ["all", "basic"]:
        print("Initializing Basic Model...")
        model_dict["Basic Model"] = BasicMedicalModel(data_path)
    
    if models_to_run.lower() in ["all", "improved"]:
        print("Initializing Improved Model...")
        model_dict["Improved Model"] = ImprovedMedicalModel(data_path)
    
    if models_to_run.lower() in ["all", "bert"]:
        print("Initializing BERT Model...")
        model_dict["BERT Model"] = BERTMedicalModel(data_path, device=device)
    
    if not model_dict:
        print(f"No valid models selected. Please choose from: all, bert, basic, improved")
        return
    
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
    results = {}
    for model_name in model_dict:
        results[model_name] = {
            'found': 0, 
            'avg_rank': [], 
            'avg_confidence': [], 
            'times': [], 
            'real_results': []
        }
    
    # Run benchmarks on random test cases
    print("Running benchmarks on random test cases...")
    for i, case in enumerate(tqdm(random_test_cases)):
        true_disease = case['true_disease']
        symptoms = case['symptoms']
        diagnosis = case['diagnosis']
        
        # Test each model
        for model_name, model in model_dict.items():
            start_time = time.time()
            result = model.predict(symptoms, diagnosis)
            elapsed_time = time.time() - start_time
            eval_result = evaluate_result(result, true_disease)
            
            # Record results
            if eval_result['found']:
                results[model_name]['found'] += 1
                results[model_name]['avg_rank'].append(eval_result['rank'])
                results[model_name]['avg_confidence'].append(eval_result['confidence'])
            results[model_name]['times'].append(elapsed_time)
    
    # Run benchmarks on real test cases
    if include_real:
        print("Running benchmarks on real-world examples...")
        for case in real_test_cases:
            true_disease = case['true_disease']
            symptoms = case['symptoms']
            diagnosis = case['diagnosis']
            
            # Test each model
            for model_name, model in model_dict.items():
                start_time = time.time()
                result = model.predict(symptoms, diagnosis)
                elapsed_time = time.time() - start_time
                eval_result = evaluate_result(result, true_disease)
                
                # Record real case results
                results[model_name]['real_results'].append({
                    'case': case['name'],
                    'found': eval_result['found'],
                    'rank': eval_result['rank'],
                    'confidence': eval_result['confidence'],
                    'time': elapsed_time
                })
    
    # Calculate final metrics for random test cases
    for model_name in results:
        results[model_name]['accuracy'] = results[model_name]['found'] / num_test_cases
        results[model_name]['avg_rank'] = np.mean(results[model_name]['avg_rank']) if results[model_name]['avg_rank'] else None
        results[model_name]['avg_confidence'] = np.mean(results[model_name]['avg_confidence']) if results[model_name]['avg_confidence'] else None
        results[model_name]['avg_time'] = np.mean(results[model_name]['times'])
    
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
    output_dir = os.path.join(args.output_dir, f"bert_benchmark_{timestamp}")
    
    # Set up logging
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "benchmark_log.txt")
    logger = TeeLogger(log_file)
    sys.stdout = logger
    
    try:
        # Print start message
        print("=" * 60)
        print(f"Medical Advisor Models Benchmark with Bio_ClinicalBERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Data file: {args.data}")
        print(f"Number of random test cases: {args.num_tests}")
        print(f"Include real-world examples: {args.real_examples}")
        print(f"Device for BERT: {args.device if args.device else 'auto'}")
        print(f"Models to benchmark: {args.models}")
        print(f"Results will be saved to: {output_dir}")
        print("=" * 60)
        
        # Run the benchmark
        results = run_benchmark(
            data_path=args.data,
            num_test_cases=args.num_tests,
            include_real=args.real_examples,
            device=args.device,
            models_to_run=args.models,
            output_dir=output_dir
        )
        
        # Print completion message
        print("\nBenchmark completed.")
        print(f"Results saved to: {output_dir}")
    
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdout and close log file
        sys.stdout = sys.__stdout__
        logger.close()
        print(f"Benchmark finished. Results saved to: {output_dir}")
        print(f"Log file: {log_file}")

if __name__ == "__main__":
    main()
