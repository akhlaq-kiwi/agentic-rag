#!/usr/bin/env python3
"""
Main evaluation runner for RAGAS-based RAG system evaluation
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ragas_evaluator import RAGASEvaluator, create_sample_test_cases, load_test_cases_from_file
from evaluation.phoenix_tracer import setup_phoenix_tracing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_services():
    """Wait for required services to be available"""
    import requests
    
    services = {
        "RAG API": os.getenv("RAG_API_BASE_URL", "http://rag-api:8000"),
        "Ollama": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        "Phoenix": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006")
    }
    
    for service_name, url in services.items():
        logger.info(f"Waiting for {service_name} at {url}")
        
        for attempt in range(30):  # Wait up to 5 minutes
            try:
                if service_name == "Phoenix":
                    # Phoenix might not have a health endpoint, just check if it's reachable
                    response = requests.get(f"{url}/", timeout=5)
                else:
                    response = requests.get(f"{url}/", timeout=5)
                
                if response.status_code < 500:
                    logger.info(f"✓ {service_name} is ready")
                    break
            except Exception as e:
                if attempt < 29:
                    logger.info(f"  Waiting for {service_name}... (attempt {attempt + 1}/30)")
                    time.sleep(10)
                else:
                    logger.warning(f"  {service_name} not ready after 5 minutes: {e}")

def main():
    """Main evaluation function"""
    logger.info("Starting RAGAS evaluation for Agentic RAG system")
    
    # Wait for services
    wait_for_services()
    
    # Setup Phoenix tracing
    setup_phoenix_tracing()
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        rag_api_url=os.getenv("RAG_API_BASE_URL", "http://rag-api:8000"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    )
    
    # Load test cases
    test_cases_file = "/app/evaluation/test_cases.json"
    if os.path.exists(test_cases_file):
        logger.info(f"Loading test cases from {test_cases_file}")
        test_cases = load_test_cases_from_file(test_cases_file)
    else:
        logger.info("Using sample test cases")
        test_cases = create_sample_test_cases()
    
    if not test_cases:
        logger.error("No test cases available for evaluation")
        return
    
    # Run evaluation
    try:
        report_path = evaluator.generate_evaluation_report(
            test_cases=test_cases,
            output_path="/app/evaluation/results"
        )
        logger.info(f"Evaluation completed successfully. Report: {report_path}")
        
        # Also run quick evaluation for metrics only
        scores = evaluator.evaluate_rag_system(test_cases)
        
        # Print summary
        print("\n" + "="*50)
        print("RAGAS EVALUATION SUMMARY")
        print("="*50)
        print(f"Total test cases: {len(test_cases)}")
        print(f"Metrics evaluated: {len(scores)}")
        print("\nMetric Scores:")
        for metric, score in scores.items():
            print(f"  {metric:20}: {score:.4f}")
        
        if scores:
            overall_score = sum(scores.values()) / len(scores)
            print(f"\nOverall Score: {overall_score:.4f}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
