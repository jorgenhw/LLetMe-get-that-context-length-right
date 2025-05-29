from typing import Dict, List, Tuple
import openai
import json
import random
from pathlib import Path
import logging
from ..data_structures import Document, Needle

logger = logging.getLogger(__name__)

class MultiNeedleTask:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_needle(self, context: str, document_style: str, existing_needles: List[str]) -> Tuple[str, str, str]:
        """Generate a context-aware needle and corresponding query"""
        
        prompt = f"""
You are tasked with creating a context-aware needle (a piece of information) for document evaluation.

Context from document:
{context}

Document style/type: {document_style}

Existing needles in document:
{json.dumps(existing_needles, indent=2)}

Requirements:
1. Create a factual statement (the needle) that:
   - Fits naturally with the document's style and content
   - Contains specific, verifiable information (e.g., dates, numbers, names, facts)
   - Is 1-2 sentences maximum
   - Could plausibly be part of this document
   - Is DIFFERENT from existing needles but maintains document coherence
2. Create a question that:
   - Can ONLY be answered using the needle information
   - Uses different wording than the needle
   - Is specific and unambiguous
3. Provide the exact answer to the question

Respond in JSON format:
{{
    "needle": "Your factual statement to insert",
    "query": "Question that can only be answered by the needle",
    "answer": "The exact answer to the query"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["needle"], result["query"], result["answer"]
            
        except Exception as e:
            logger.error(f"Error generating needle: {str(e)}")
            raise

    def determine_insertion_points(self, document: Document, num_needles: int) -> List[Tuple[int, str, float]]:
        """Determine where to insert multiple needles in the document"""
        
        # Calculate total document length
        total_length = sum(len(page.split()) for page in document.pages)
        
        # Create insertion points ensuring good distribution
        insertion_points = []
        used_positions = set()
        
        for _ in range(num_needles):
            while True:
                # Randomly choose position bin (front, middle, back)
                position_bin = random.choice(["front", "middle", "back"])
                
                if position_bin == "front":
                    target_page = random.randint(0, len(document.pages) // 3)
                    position_percent = random.uniform(0, 33)
                elif position_bin == "middle":
                    target_page = random.randint(len(document.pages) // 3, 2 * len(document.pages) // 3)
                    position_percent = random.uniform(33, 66)
                else:  # back
                    target_page = random.randint(2 * len(document.pages) // 3, len(document.pages) - 1)
                    position_percent = random.uniform(66, 100)
                
                # Check if position is unique enough (not too close to existing positions)
                position_key = (target_page, int(position_percent / 10))  # Group by 10% chunks
                if position_key not in used_positions:
                    used_positions.add(position_key)
                    insertion_points.append((target_page, position_bin, position_percent))
                    break
            
        return insertion_points

    def insert_needles(self, document: Document, num_needles: int) -> Tuple[Document, List[Needle]]:
        """Insert multiple context-aware needles into the document"""
        
        # Determine insertion points
        insertion_points = self.determine_insertion_points(document, num_needles)
        needles = []
        existing_needle_texts = []
        
        for target_page, position_bin, position_percent in insertion_points:
            page_content = document.pages[target_page]
            
            # Generate needle based on surrounding context
            needle_text, query, expected_answer = self.generate_needle(
                context=page_content,
                document_style="formal document",
                existing_needles=existing_needle_texts
            )
            existing_needle_texts.append(needle_text)
            
            # Insert needle into the page
            paragraphs = page_content.split('\n\n')
            insert_idx = int(len(paragraphs) * (position_percent / 100))
            paragraphs.insert(insert_idx, needle_text)
            
            # Update document
            document.pages[target_page] = '\n\n'.join(paragraphs)
            
            # Create needle object
            needle = Needle(
                content=needle_text,
                page_number=target_page + 1,
                position_percent=position_percent,
                insertion_point=f"page {target_page + 1}, {position_bin} section",
                query=query,
                expected_answer=expected_answer,
                task_type="multi_needle",
                num_needles=num_needles
            )
            needles.append(needle)
        
        # Update full document content
        document.content = '\n'.join(document.pages)
        document.needles.extend(needles)
        
        return document, needles

    def evaluate_response(self, model_answer: str, expected_answer: str) -> Tuple[bool, float, str]:
        """Evaluate the model's response using GPT-4"""
        
        eval_prompt = f"""
Evaluate if the model's answer matches the expected answer.

Expected Answer: {expected_answer}
Model Answer: {model_answer}

Requirements:
- Exact factual match is required
- Minor wording differences are acceptable
- Answer must contain all key information

Respond in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation of the evaluation"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["is_correct"], result["confidence"], result["explanation"]
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            raise

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate accuracy for multi-needle task"""
        
        total_needles = len(results)
        correct_answers = sum(1 for r in results if r["is_correct"])
        
        accuracy = correct_answers / total_needles if total_needles > 0 else 0
        
        return {
            "accuracy": accuracy
        }

    def run(self, document: Document, model_inference) -> List[Dict]:
        """Run the multi-needle task on a document"""
        
        results = []
        
        try:
            # Randomly choose number of needles (2-5)
            num_needles = random.randint(2, 5)
            
            # Insert needles
            document, needles = self.insert_needles(document, num_needles)
            
            # Evaluate each needle
            needle_results = []
            for needle in needles:
                # Get model's answer
                model_answer = model_inference.query_model(document, needle.query)
                
                # Evaluate answer
                is_correct, confidence, explanation = self.evaluate_response(
                    model_answer=model_answer,
                    expected_answer=needle.expected_answer
                )
                
                needle_results.append({
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "explanation": explanation,
                    "model_answer": model_answer,
                    "query": needle.query,
                    "expected_answer": needle.expected_answer,
                    "needle_text": needle.content
                })
            
            # Calculate metrics
            metrics = self.calculate_metrics(needle_results)
            
            # Record results
            results.append({
                "model_name": model_inference.model_name,
                "document_id": document.id,
                "task_type": "multi_needle",
                "metric_type": "accuracy",
                "value": float(metrics["accuracy"]),
                "metadata": {
                    "num_needles": num_needles,
                    "needle_results": needle_results,
                    "total_needles": len(needle_results),
                    "correct_needles": sum(1 for r in needle_results if r["is_correct"])
                }
            })
            
        except Exception as e:
            logger.error(f"Error running multi needle task: {str(e)}")
            raise
            
        return results 