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

def setup_results_directory():
    """Create results directory if it doesn't exist."""
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def save_evaluation_results(results, dataset_name: str, results_dir: Path):
    """Save evaluation results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ragas_evaluation_{dataset_name.replace('.jsonl', '')}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Convert results to dict if needed
    if hasattr(results, 'to_dict'):
        results_dict = results.to_dict()
    elif hasattr(results, '__dict__'):
        results_dict = results.__dict__
    else:
        results_dict = {"results": str(results)}
    
    # Add metadata
    results_dict["metadata"] = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "evaluation_date": datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"📊 Results saved to: {filepath}")
    return filepath

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
        
        # Save results if requested
        if args.save_results:
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