import json
from typing import List, Dict, Tuple
import pandas as pd
import openai
import logging
from .data_structures import EvaluationResult, Document

logger = logging.getLogger(__name__)

class LLMEvaluator:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def evaluate_answer(self, expected_answer: str, model_answer: str, task_type: str) -> Tuple[int, bool, Dict]:
        """Evaluate model answer using LLM"""
        
        eval_prompt = f"""
You are evaluating the correctness of an AI model's answer.

Expected Answer: {expected_answer}
Model Answer: {model_answer}
Task Type: {task_type}

Evaluation Criteria for different tasks:

Single/Multi Needle:
- Score 2: Completely correct and accurate
- Score 1: Partially correct (some elements right, some missing/wrong)
- Score 0: Incorrect or no relevant information

Multi-Hop:
- Track each reasoning step
- Identify where the chain breaks (if it does)
- Classify errors as retrieval or logic errors

Aggregation:
- Check completeness of component retrieval
- Verify correctness of synthesis
- Evaluate quality of combined result

Respond in JSON format:
{{
    "score": 0/1/2,
    "is_correct": true/false,
    "error_type": "none/retrieval/logic/synthesis",
    "completeness": 0.0-1.0,
    "correctness": 0.0-1.0,
    "explanation": "Brief explanation"
}}
"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["score"], result["is_correct"], {
            "error_type": result.get("error_type", "none"),
            "completeness": result.get("completeness", 1.0),
            "correctness": result.get("correctness", 1.0)
        }

class EvaluationFramework:
    def __init__(self, openai_api_key: str):
        logger.info("Initializing EvaluationFramework")
        self.evaluator = LLMEvaluator(openai_api_key)
        self.results: List[Dict] = []
    
    def run_evaluation(self, documents: List[Document], model_inference, model_name: str):
        """Run complete evaluation on all documents and tasks"""
        
        for document in documents:
            for needle in document.needles:
                # Get model response
                model_answer = model_inference.query_model(document, needle.query)
                
                # Evaluate response
                score, is_correct, eval_metrics = self.evaluator.evaluate_answer(
                    needle.expected_answer, 
                    model_answer, 
                    needle.task_type
                )
                
                # Calculate position segment
                position = needle.position_percent
                if position < 33:
                    position_segment = "beginning"
                elif position < 66:
                    position_segment = "middle"
                else:
                    position_segment = "end"
                
                # Store result with enhanced metadata
                result = {
                    "model_name": model_name,
                    "document_id": document.id,
                    "task_type": needle.task_type,
                    "query": needle.query,
                    "model_answer": model_answer,
                    "expected_answer": needle.expected_answer,
                    "score": score,
                    "is_correct": is_correct,
                    "metadata": {
                        "page_number": needle.page_number,
                        "position_percent": needle.position_percent,
                        "position_segment": position_segment,
                        "document_length": document.length_tokens,
                        "error_type": eval_metrics["error_type"],
                        "completeness": eval_metrics["completeness"],
                        "correctness": eval_metrics["correctness"]
                    }
                }
                
                self.results.append(result)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        if not self.results:
            logger.warning("No results to calculate metrics from")
            return {}
        
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            metrics = {}
            
            # Single Needle Metrics
            single_df = df[df["task_type"] == "single_needle"]
            if not single_df.empty:
                for pos in ["beginning", "middle", "end"]:
                    pos_df = single_df[single_df["position_segment"] == pos]
                    metrics[f"single_needle_accuracy_{pos}"] = pos_df["is_correct"].mean()
            
            # Multi Needle Metrics
            multi_df = df[df["task_type"] == "multi_needle"]
            if not multi_df.empty:
                metrics.update({
                    "multi_needle_precision": len(multi_df[multi_df["is_correct"]]) / len(multi_df),
                    "multi_needle_recall": multi_df["completeness"].mean(),
                    "multi_needle_f1": 2 * (metrics["multi_needle_precision"] * metrics["multi_needle_recall"]) / 
                                     (metrics["multi_needle_precision"] + metrics["multi_needle_recall"]),
                    "multi_needle_partial": len(multi_df[multi_df["score"] >= 1]) / len(multi_df)
                })
            
            # Multi Hop Metrics
            hop_df = df[df["task_type"] == "multi_hop"]
            if not hop_df.empty:
                for hop_count in hop_df["hop_count"].unique():
                    hop_specific = hop_df[hop_df["hop_count"] == hop_count]
                    metrics[f"multi_hop_success_{hop_count}"] = hop_specific["is_correct"].mean()
                
                max_hops = hop_df["hop_count"].max()
                metrics["hop_decay"] = -(
                    hop_df[hop_df["hop_count"] == max_hops]["is_correct"].mean() -
                    hop_df[hop_df["hop_count"] == 1]["is_correct"].mean()
                ) / (max_hops - 1)
                
                # Error analysis
                metrics.update({
                    "retrieval_error_rate": len(hop_df[hop_df["error_type"] == "retrieval"]) / len(hop_df),
                    "logic_error_rate": len(hop_df[hop_df["error_type"] == "logic"]) / len(hop_df)
                })
            
            # Aggregation Metrics
            agg_df = df[df["task_type"] == "aggregation"]
            if not agg_df.empty:
                metrics.update({
                    "aggregation_completeness": agg_df["completeness"].mean(),
                    "aggregation_correctness": agg_df["correctness"].mean(),
                    "aggregation_synthesis": agg_df["is_correct"].mean()
                })
            
            # Document length impact
            df["length_bin"] = pd.qcut(df["document_length"], q=3, labels=["short", "medium", "long"])
            length_impact = df.groupby("length_bin")["is_correct"].mean()
            metrics["length_impact"] = length_impact.to_dict()
            
            logger.info("Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {}
    
    def save_results(self, filepath: str):
        """Save results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(self.results)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            raise
