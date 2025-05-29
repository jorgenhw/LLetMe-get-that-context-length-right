from typing import Dict, List, Tuple
import openai
import json
import random
from pathlib import Path
import logging
from ..data_structures import Document, Needle

logger = logging.getLogger(__name__)

class SingleNeedleTask:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_needle(self, context: str, document_style: str) -> Tuple[str, str, str]:
        """Generate a context-aware needle and corresponding query"""
        
        prompt = f"""
You are tasked with creating a context-aware needle (a piece of information) for document evaluation.

Context from document:
{context}

Document style/type: {document_style}

Requirements:
1. Create a factual statement (the needle) that:
   - Fits naturally with the document's style and content
   - Contains specific, verifiable information (e.g., dates, numbers, names, facts)
   - Is 1-2 sentences maximum
   - Could plausibly be part of this document
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

    def determine_insertion_point(self, document: Document) -> Tuple[int, str, float]:
        """Determine where to insert the needle in the document"""
        
        # Calculate total document length
        total_length = sum(len(page.split()) for page in document.pages)
        
        # Randomly choose position bin (front, middle, back)
        position_bin = random.choice(["front", "middle", "back"])
        
        if position_bin == "front":
            target_page = 0
            position_percent = random.uniform(0, 33)
        elif position_bin == "middle":
            target_page = len(document.pages) // 2
            position_percent = random.uniform(33, 66)
        else:  # back
            target_page = len(document.pages) - 1
            position_percent = random.uniform(66, 100)
            
        return target_page, position_bin, position_percent

    def insert_needle(self, document: Document) -> Tuple[Document, Needle]:
        """Insert a single context-aware needle into the document"""
        
        # Determine insertion point
        target_page, position_bin, position_percent = self.determine_insertion_point(document)
        page_content = document.pages[target_page]
        
        # Generate needle based on surrounding context
        needle_text, query, expected_answer = self.generate_needle(
            context=page_content,
            document_style="formal document"
        )
        
        # Insert needle into the page
        paragraphs = page_content.split('\n\n')
        insert_idx = int(len(paragraphs) * (position_percent / 100))
        paragraphs.insert(insert_idx, needle_text)
        
        # Update document
        document.pages[target_page] = '\n\n'.join(paragraphs)
        document.content = '\n'.join(document.pages)
        
        # Create needle object
        needle = Needle(
            content=needle_text,
            page_number=target_page + 1,
            position_percent=position_percent,
            insertion_point=f"page {target_page + 1}, {position_bin} section",
            query=query,
            expected_answer=expected_answer,
            task_type="single_needle"
        )
        
        document.needles.append(needle)
        return document, needle

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

    def run(self, document: Document, model_inference) -> List[Dict]:
        """Run the single needle task on a document"""
        
        results = []
        
        try:
            # Insert needle
            document, needle = self.insert_needle(document)
            
            # Get model's answer
            model_answer = model_inference.query_model(document, needle.query)
            
            # Evaluate answer
            is_correct, confidence, explanation = self.evaluate_response(
                model_answer=model_answer,
                expected_answer=needle.expected_answer
            )
            
            # Record results
            results.append({
                "model_name": model_inference.model_name,
                "document_id": document.id,
                "task_type": "single_needle",
                "metric_type": "accuracy",
                "value": float(is_correct),
                "metadata": {
                    "position": needle.insertion_point.split(", ")[1].split(" ")[0],
                    "confidence": confidence,
                    "explanation": explanation,
                    "query": needle.query,
                    "expected_answer": needle.expected_answer,
                    "model_answer": model_answer,
                    "needle_text": needle.content,
                    "page_number": needle.page_number,
                    "position_percent": needle.position_percent
                }
            })
            
        except Exception as e:
            logger.error(f"Error running single needle task: {str(e)}")
            raise
            
        return results 