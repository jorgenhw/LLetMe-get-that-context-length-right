import argparse
from src.pipeline import EvaluationPipeline
from src.model_inference import OpenAIInference
from dotenv import load_dotenv
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.debug("Environment variables loaded from .env file")

def main():
    logger.info("Starting evaluation pipeline")
    
    # Create pdf_files directory if it doesn't exist
    pdf_dir = Path("pdf_files")
    pdf_dir.mkdir(exist_ok=True)
    
    # Get all PDF files from the pdf_files directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in the pdf_files directory")
        print("Error: No PDF files found in the pdf_files directory.")
        print("Please add some PDF files to the pdf_files directory and try again.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files in pdf_files directory")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run long-context LLM evaluation')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode without requiring API key')
    parser.add_argument('--no-anonymize', action='store_true', help='Disable document anonymization (enabled by default)')
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key and not args.mock:
        logger.error("OPENAI_API_KEY not found in environment variables")
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key or use --mock mode")
        return
    elif args.mock:
        logger.info("Running in mock mode - API key not required")
        api_key = "mock-key"
    else:
        logger.debug("API key successfully retrieved from environment")

    # Initialize models
    logger.info("Initializing models")
    models = {
        'gpt-4o': OpenAIInference(api_key=api_key, model_name='gpt-4o', mock=args.mock),
    }
    logger.debug(f"Models initialized: {list(models.keys())}")

    # Initialize and run pipeline
    logger.info("Initializing evaluation pipeline")
    pipeline = EvaluationPipeline(models, api_key, mock=args.mock, should_anonymize=not args.no_anonymize)
    
    # Convert Path objects to strings for the pipeline
    pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
    logger.info("Starting evaluation with PDF files: %s", pdf_paths)
    
    # Run evaluation for both tasks
    results = pipeline.run_evaluation(pdf_paths, task_type="all")
    logger.info("Evaluation completed successfully")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Print summary of results
    single_needle_results = [r for r in results if r["task_type"] == "single_needle"]
    multi_needle_results = [r for r in results if r["task_type"] == "multi_needle"]
    
    print("\nEvaluation Results Summary:")
    print("==========================")
    
    if single_needle_results:
        print("\nSingle Needle Task:")
        print("-----------------")
        accuracy = sum(1 for r in single_needle_results if r.get("is_correct", False)) / len(single_needle_results)
        print(f"Total needles: {len(single_needle_results)}")
        print(f"Accuracy: {accuracy:.2%}")
    
    if multi_needle_results:
        print("\nMulti Needle Task:")
        print("----------------")
        metrics = next((r for r in multi_needle_results if r["metric_type"] == "accuracy"), {})
        if metrics:
            total_needles = metrics.get("metadata", {}).get("total_needles", 0)
            correct_needles = metrics.get("metadata", {}).get("correct_needles", 0)
            print(f"Total documents: {len(set(r['document_id'] for r in multi_needle_results))}")
            print(f"Total needles: {total_needles}")
            print(f"Correct needles: {correct_needles}")
            print(f"Accuracy: {metrics.get('value', 0):.2%}")
    
    logger.info("Results summary printed to console")

if __name__ == '__main__':
    try:
        main()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}", exc_info=True)
        raise
