"""
RAGAS Evaluation Module for Agentic RAG System
Evaluates RAG performance using RAGAS metrics
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from datasets import Dataset
import requests
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGASEvaluator:
    """RAGAS-based evaluator for RAG system performance"""
    
    def __init__(self, 
                 rag_api_url: str = "http://rag-api:8000",
                 ollama_url: str = "http://ollama:11434"):
        self.rag_api_url = rag_api_url
        self.ollama_url = ollama_url
        
        # Initialize LLM and embeddings for RAGAS
        self.llm = Ollama(
            model="gemma:2b",
            base_url=ollama_url
        )
        
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=ollama_url
        )
        
        # RAGAS metrics to evaluate
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        ]
    
    def query_rag_system(self, question: str) -> Dict[str, Any]:
        """Query the RAG system and return response with context"""
        try:
            response = requests.get(
                f"{self.rag_api_url}/query",
                params={"question": question},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {"answer": "Error", "context": []}
    
    def prepare_evaluation_dataset(self, test_cases: List[Dict]) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation
        
        Expected test_case format:
        {
            "question": "What is procurement framework?",
            "ground_truth": "Expected answer...",
            "contexts": ["Retrieved context 1", "Retrieved context 2"]
        }
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for test_case in test_cases:
            question = test_case["question"]
            ground_truth = test_case.get("ground_truth", "")
            
            # Query RAG system
            rag_response = self.query_rag_system(question)
            answer = rag_response.get("answer", "")
            
            # Extract contexts (assuming they're in the response)
            context_list = test_case.get("contexts", [])
            if not context_list and "context" in rag_response:
                context_list = [rag_response["context"]]
            
            questions.append(question)
            answers.append(answer)
            contexts.append(context_list)
            ground_truths.append(ground_truth)
        
        # Create RAGAS dataset
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def evaluate_rag_system(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate RAG system using RAGAS metrics"""
        logger.info(f"Evaluating RAG system with {len(test_cases)} test cases")
        
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(test_cases)
        
        # Run RAGAS evaluation
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Convert to dictionary
            scores = {}
            for metric_name, score in result.items():
                if isinstance(score, (int, float)):
                    scores[metric_name] = float(score)
                else:
                    scores[metric_name] = float(score.mean()) if hasattr(score, 'mean') else 0.0
            
            logger.info("RAGAS Evaluation Results:")
            for metric, score in scores.items():
                logger.info(f"  {metric}: {score:.4f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            return {}
    
    def generate_evaluation_report(self, 
                                 test_cases: List[Dict], 
                                 output_path: str = "/app/evaluation/results") -> str:
        """Generate comprehensive evaluation report"""
        
        # Run evaluation
        scores = self.evaluate_rag_system(test_cases)
        
        # Create detailed report
        report = {
            "evaluation_summary": {
                "total_test_cases": len(test_cases),
                "metrics_evaluated": list(scores.keys()),
                "overall_score": sum(scores.values()) / len(scores) if scores else 0.0
            },
            "metric_scores": scores,
            "test_cases": []
        }
        
        # Add individual test case results
        for i, test_case in enumerate(test_cases):
            rag_response = self.query_rag_system(test_case["question"])
            
            case_result = {
                "test_case_id": i + 1,
                "question": test_case["question"],
                "expected_answer": test_case.get("ground_truth", ""),
                "rag_answer": rag_response.get("answer", ""),
                "contexts_retrieved": test_case.get("contexts", [])
            }
            report["test_cases"].append(case_result)
        
        # Save report
        os.makedirs(output_path, exist_ok=True)
        report_file = os.path.join(output_path, "ragas_evaluation_report.json")
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {report_file}")
        return report_file

def load_test_cases_from_file(file_path: str) -> List[Dict]:
    """Load test cases from JSON file"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading test cases: {e}")
        return []

def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases for evaluation"""
    return [
        {
            "question": "What is the procurement framework?",
            "ground_truth": "The Procurement Framework is a comprehensive set of guidelines and procedures that define the responsibilities and processes of all procurement activities within an organization.",
            "contexts": [
                "The procurement framework establishes clear guidelines for purchasing activities.",
                "It includes supplier management, contract management, and procurement planning processes."
            ]
        },
        {
            "question": "What are the key components of information security?",
            "ground_truth": "Key components of information security include confidentiality, integrity, availability, authentication, authorization, and non-repudiation.",
            "contexts": [
                "Information security focuses on protecting data confidentiality, integrity, and availability.",
                "Authentication and authorization are critical security controls."
            ]
        },
        {
            "question": "What are HR bylaws?",
            "ground_truth": "HR bylaws are formal rules and regulations that govern human resources policies, procedures, and employee conduct within an organization.",
            "contexts": [
                "HR bylaws establish employee policies and procedures.",
                "They define organizational rules for human resources management."
            ]
        }
    ]
