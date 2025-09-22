import os
import sys
import asyncio
from typing import Dict, List, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv
from src.rag.crew import run_rag_query
from src.config import (EVALUATION_DATA_PATH)


class RagasEvaluator:
    """
    A class-based evaluator for RAG systems using RAGAS metrics.
    
    This class encapsulates all the logic for evaluating a RAG pipeline
    using the RAGAS framework, including dataset loading, pipeline execution,
    and metric evaluation.
    """

    def __init__(self, dataset: str = None):
        """
        Initialize the RagasEvaluator with configuration.
        
        Args:
            eval_dataset_path (str, optional): Path to the evaluation dataset.
                                             If None, uses default path.
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Set up paths
        self.eval_dataset_path =f"{EVALUATION_DATA_PATH}/{dataset or 'small_dataset.jsonl'}"
        
        # Define the metrics we want to use
        self.metrics = [
            faithfulness,       # How factually accurate is the answer based on the context?
            answer_relevancy,   # How relevant is the answer to the question?
            context_recall,     # Did the retriever find all the relevant context?
            context_precision,  # Was the retrieved context precise and not full of noise?
        ]

    def run_rag_pipeline(self, query: str) -> Dict[str, Any]:
        """
        A wrapper method to run the CrewAI RAG pipeline and return the final result.
        
        Args:
            query (str): The question to be processed by the RAG pipeline.
            
        Returns:
            Dict[str, Any]: Dictionary containing 'answer' and 'contexts' keys.
        """
        try:
            result = run_rag_query(query)
            answer_string = str(result)
            answer = answer_string
            contexts = [answer_string]  # Use the string version here as well

            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            print(f"Error running crew for query '{query}': {e}")
            return {"answer": "Error", "contexts": []}

    def load_dataset(self) -> Dataset:
        """
        Load the evaluation dataset from the specified JSON Lines file.
        
        Returns:
            Dataset: The loaded evaluation dataset.
            
        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
        """
        print(f"📚 Loading evaluation dataset from: {self.eval_dataset_path}")
        if not os.path.exists(self.eval_dataset_path):
            raise FileNotFoundError(f"❌ Error: Evaluation dataset not found at {self.eval_dataset_path}")
        
        return Dataset.from_json(self.eval_dataset_path)
    
    def process_dataset(self, golden_dataset: Dataset) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Process the golden dataset by running the RAG pipeline on each question.
        
        Args:
            golden_dataset (Dataset): The golden evaluation dataset.
            
        Returns:
            tuple: A tuple containing (questions, results, ground_truths).
        """
        print("\n🚀 Running RAG pipeline on the evaluation dataset...")
        results = []
        questions = []
        ground_truths = []
        
        for entry in golden_dataset:
            question = entry['question']
            ground_truth = entry['ground_truth']
            
            print(f"  - Processing question: '{question[:80]}...'")
            pipeline_output = self.run_rag_pipeline(question)
            
            results.append(pipeline_output)
            questions.append(question)
            ground_truths.append(ground_truth)
        
        return questions, results, ground_truths
    
    def create_evaluation_dataset(self, questions: List[str], results: List[Dict[str, Any]], ground_truths: List[str]) -> Dataset:
        """
        Create the evaluation dataset for RAGAS.
        
        Args:
            questions (List[str]): List of questions.
            results (List[Dict[str, Any]]): List of pipeline results.
            ground_truths (List[str]): List of ground truth answers.
            
        Returns:
            Dataset: The formatted evaluation dataset.
        """
        evaluation_data = {
            "question": questions,
            "answer": [res["answer"] for res in results],
            "contexts": [res["contexts"] for res in results],
            "ground_truth": ground_truths,
        }
        return Dataset.from_dict(evaluation_data)
    
    def evaluate_with_ragas(self, eval_dataset: Dataset) -> Any:
        """
        Run the RAGAS evaluation on the prepared dataset.
        
        Args:
            eval_dataset (Dataset): The evaluation dataset.
            
        Returns:
            Any: The RAGAS evaluation results.
        """
        print("Evaluating the results with RAGAS...")
        
        # Run the evaluation
        result = evaluate(
            dataset=eval_dataset,
            metrics=self.metrics,
        )
        
        return result
    
    async def run_evaluation(self) -> Any:
        """
        Main method to run the complete RAGAS evaluation pipeline.
        
        Returns:
            Any: The evaluation results.
        """
        try:
            # Load the dataset
            golden_dataset = self.load_dataset()
            
            # Process the dataset through RAG pipeline
            questions, results, ground_truths = self.process_dataset(golden_dataset)
            
            # Create evaluation dataset
            eval_dataset = self.create_evaluation_dataset(questions, results, ground_truths)
            
            # Run RAGAS evaluation
            result = self.evaluate_with_ragas(eval_dataset)
            
            # Display results
            print("\n✅ Evaluation Complete!")
            print("-------------------------")
            print(result)
            print("-------------------------")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise

async def raga_runner(dataset: str):
    """
    Main function to initialize and run the RAGAS evaluation using the class-based approach.
    """
    # Initialize the evaluator
    evaluator = RagasEvaluator(dataset) 
    # Run the evaluation
    await evaluator.run_evaluation()
