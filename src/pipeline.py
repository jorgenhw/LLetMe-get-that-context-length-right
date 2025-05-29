import random
from pathlib import Path
from typing import List, Dict
import logging
from .document_processor import DocumentProcessor
from .tasks import SingleNeedleTask, MultiNeedleTask, MultiHopTask, AggregationTask
from .evaluation import EvaluationFramework
from .model_inference import ModelInference
from .data_structures import Document

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    def __init__(self, models: Dict[str, ModelInference], openai_api_key: str, mock: bool = False, should_anonymize: bool = True):
        logger.info("Initializing EvaluationPipeline")
        self.models = models
        self.document_processor = DocumentProcessor(should_anonymize=should_anonymize)
        self.single_needle_task = SingleNeedleTask(openai_api_key)
        self.multi_needle_task = MultiNeedleTask(openai_api_key)
        self.multi_hop_task = MultiHopTask(openai_api_key)
        self.aggregation_task = AggregationTask(openai_api_key)
        self.evaluation_framework = EvaluationFramework(openai_api_key)
        logger.debug(f"Initialized with {len(models)} models")
        logger.debug(f"Document anonymization is {'enabled' if should_anonymize else 'disabled'}")

    def run_evaluation(self, pdf_paths: List[str], task_type: str = "all") -> List[Dict]:
        """Run evaluation pipeline on provided PDF files"""
        
        results = []
        
        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing {pdf_path}")
                
                # Process document
                doc_id = Path(pdf_path).stem 
                document = self.document_processor.create_document(pdf_path, doc_id)
                
                # Run tasks for each model
                for model_name, model in self.models.items():
                    logger.info(f"Running evaluation with model: {model_name}")
                    
                    # Run single needle task
                    if task_type in ["all", "single_needle"]:
                        try:
                            single_needle_results = self.single_needle_task.run(document, model)
                            results.extend(single_needle_results)
                        except Exception as e:
                            logger.error(f"Error in single needle task: {str(e)}")
                    
                    # Run multi needle task
                    if task_type in ["all", "multi_needle"]:
                        try:
                            multi_needle_results = self.multi_needle_task.run(document, model)
                            results.extend(multi_needle_results)
                        except Exception as e:
                            logger.error(f"Error in multi needle task: {str(e)}")
                    
                    # Run multi hop task
                    if task_type in ["all", "multi_hop"]:
                        try:
                            multi_hop_results = self.multi_hop_task.run(document, model)
                            results.extend(multi_hop_results)
                        except Exception as e:
                            logger.error(f"Error in multi hop task: {str(e)}")
                    
                    # Run aggregation task
                    if task_type in ["all", "aggregation"]:
                        try:
                            aggregation_results = self.aggregation_task.run(document, model)
                            results.extend(aggregation_results)
                        except Exception as e:
                            logger.error(f"Error in aggregation task: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                raise
        
        # Save results to separate CSV files
        if results:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Group results by task type
            single_results = [r for r in results if r["task_type"] == "single_needle"]
            multi_results = [r for r in results if r["task_type"] == "multi_needle"]
            hop_results = [r for r in results if r["task_type"] == "multi_hop"]
            aggregation_results = [r for r in results if r["task_type"] == "aggregation"]
            
            # Save each task's results to its own CSV
            if single_results:
                self.evaluation_framework.results = single_results
                self.evaluation_framework.save_results("results/single_needle_results.csv")
            
            if multi_results:
                self.evaluation_framework.results = multi_results
                self.evaluation_framework.save_results("results/multi_needle_results.csv")
            
            if hop_results:
                self.evaluation_framework.results = hop_results
                self.evaluation_framework.save_results("results/multi_hop_results.csv")
            
            if aggregation_results:
                self.evaluation_framework.results = aggregation_results
                self.evaluation_framework.save_results("results/aggregation_results.csv")
        
        logger.info("Evaluation pipeline completed")
        return results
