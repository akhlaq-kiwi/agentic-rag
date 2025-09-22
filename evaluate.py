#!/usr/bin/env python3
"""
RAGAS Evaluation Runner

This script runs RAGAS evaluation on the RAG system using the specified dataset.
Can be run with different datasets and saves results to evaluation_results directory.
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from src.evaluation.ragas import raga_runner
from src.config import EVALUATION_RESULTS_PATH
import csv

def setup_results_directory():
    """Create results directory if it doesn't exist."""
    results_dir = Path(f"{EVALUATION_RESULTS_PATH}")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def save_evaluation_results(results, dataset_name: str, results_dir: Path):
    """Save evaluation results to CSV file, appending each run."""
    csv_filename = "results.csv"
    csv_filepath = results_dir / csv_filename
    # print(results)
    # # Convert results to dict if needed
    # if hasattr(results, 'to_dict'):
    #     results_dict = results.to_dict()
    # elif hasattr(results, '__dict__'):
    #     results_dict = results.__dict__
    # else:
    #     results_dict = {"results": str(results)}

    # Add metadata
    results["dataset"] = dataset_name
    results["evaluation_date"] = datetime.now().isoformat()

    # Prepare row for CSV
    row = results

    # Write header only if file does not exist
    write_header = not csv_filepath.exists()
    with open(csv_filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"📊 Results appended to: {csv_filepath}")
    return csv_filepath

async def main():
    """Main function to run RAGAS evaluation with command line arguments."""
    parser = argparse.ArgumentParser(description='Run RAGAS evaluation on RAG system')
    parser.add_argument(
        '--dataset', 
        type=str, 
        default=os.getenv('EVALUATION_DATASET', 'small_dataset.jsonl'),
        help='Dataset file name (default: small_dataset.jsonl)'
    )
    parser.add_argument(
        '--save-results', 
        action='store_true', 
        default=True,
        help='Save results to file (default: True)'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting RAGAS evaluation with dataset: {args.dataset}")
    print(f"📅 Evaluation started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    try:
        # Run evaluation
        results = await raga_runner(args.dataset)
        
        results_dir = setup_results_directory()
        save_evaluation_results(results, args.dataset, results_dir)
    
        print("=" * 60)
        print("✅ Evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ragas evaluation uses asyncio, so we run the main function in an event loop
    asyncio.run(main())