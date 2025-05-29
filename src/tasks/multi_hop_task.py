from typing import Dict, List, Tuple
import openai
import json
import random
from pathlib import Path
import logging
from ..data_structures import Document, Needle

logger = logging.getLogger(__name__)

class MultiHopTask:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_hop_chain(self, context: str, document_style: str, num_hops: int) -> Tuple[List[str], str, str]:
        """Generate a chain of statements and a question that requires following the chain"""
        
        # Get a summary of the document by taking the first 1000 words
        context_summary = ' '.join(context.split()[:1000])
        
        prompt = f"""
You are tasked with creating a multi-hop reasoning chain for document evaluation.

Context from document (first 1000 words):
{context_summary}

Document style/type: {document_style}

Requirements:
1. Create {num_hops} related statements that:
   - Are realistic for legal contracts
   - Each contains specific, verifiable information
   - Form a clear logical chain
   - Use simple arithmetic if needed (avoid complex calculations)
   - Could plausibly be part of this document
2. Create a question that:
   - Can ONLY be answered by combining information from ALL statements
   - Is specific and unambiguous
   - Requires following the logical chain
3. Provide the exact answer that requires using all statements

Respond in JSON format:
{{
    "statements": [
        "Statement 1",
        "Statement 2",
        ...
    ],
    "question": "Question that requires using all statements",
    "answer": "The exact answer to the question",
    "explanation": "Step by step explanation of how to arrive at the answer"
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["statements"], result["question"], result["answer"]
            
        except Exception as e:
            logger.error(f"Error generating hop chain: {str(e)}")
            raise

    def insert_statements(self, document: Document, num_hops: int) -> Tuple[Document, List[str], str, str]:
        """Insert multi-hop statements into different pages of the document"""
        
        # Get total number of pages
        num_pages = len(document.pages)
        if num_pages < num_hops:
            raise ValueError(f"Document has fewer pages ({num_pages}) than required hops ({num_hops})")
        
        # Generate the hop chain
        statements, question, answer = self.generate_hop_chain(
            context=document.content,
            document_style="formal document",
            num_hops=num_hops
        )
        
        # Randomly select pages for each statement
        available_pages = list(range(num_pages))
        random.shuffle(available_pages)
        selected_pages = available_pages[:num_hops]
        
        # Insert each statement into a different page
        for i, (statement, page_idx) in enumerate(zip(statements, selected_pages)):
            page_content = document.pages[page_idx]
            
            # Insert statement into random position in page
            paragraphs = page_content.split('\n\n')
            insert_idx = random.randint(0, len(paragraphs))
            paragraphs.insert(insert_idx, statement)
            
            # Update document
            document.pages[page_idx] = '\n\n'.join(paragraphs)
            
            logger.debug(f"Inserted statement {i+1} on page {page_idx+1}")
        
        # Update full document content
        document.content = '\n'.join(document.pages)
        
        return document, statements, question, answer

    def evaluate_response(self, model_answer: str, expected_answer: str) -> Tuple[bool, float, str]:
        """Evaluate if the model's answer is correct"""
        
        eval_prompt = f"""
Evaluate if the model's answer matches the expected answer.

Expected Answer: {expected_answer}
Model Answer: {model_answer}

Requirements:
- Exact factual match is required
- Minor wording differences are acceptable
- Answer must demonstrate correct reasoning through all hops

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
        """Calculate success rate by hop count and hop decay"""
        
        # Group results by number of hops
        hop_results = {}
        for r in results:
            num_hops = r["metadata"]["num_hops"]
            if num_hops not in hop_results:
                hop_results[num_hops] = {"correct": 0, "total": 0}
            
            hop_results[num_hops]["total"] += 1
            if r["is_correct"]:
                hop_results[num_hops]["correct"] += 1
        
        # Calculate success rate for each hop count
        success_rates = {}
        for hops, counts in hop_results.items():
            success_rates[hops] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        
        # Calculate hop decay
        if len(success_rates) > 1:
            max_hops = max(success_rates.keys())
            hop_decay = -(success_rates[max_hops] - success_rates[1]) / (max_hops - 1)
        else:
            hop_decay = 0
        
        return {
            "success_rates": success_rates,
            "hop_decay": hop_decay
        }

    def run(self, document: Document, model_inference) -> List[Dict]:
        """Run the multi-hop task on a document"""
        
        results = []
        
        try:
            # Create a fresh copy of the document
            doc_copy = Document(
                id=document.id,  # Keep the original document ID
                content=document.content,
                pages=document.pages.copy(),
                length_tokens=document.length_tokens,
                length_pages=document.length_pages,
                needles=document.needles.copy() if document.needles else []
            )
            
            # Randomly choose number of hops (2-4)
            num_hops = random.randint(2, 4)
            
            # Insert statements and get question
            modified_doc, statements, question, expected_answer = self.insert_statements(doc_copy, num_hops)
            
            # Get model's answer
            model_answer = model_inference.query_model(modified_doc, question)
            
            # Evaluate answer
            is_correct, confidence, explanation = self.evaluate_response(
                model_answer=model_answer,
                expected_answer=expected_answer
            )
            
            # Record results
            results.append({
                "model_name": model_inference.model_name,
                "document_id": document.id,  # Use the original document ID
                "task_type": "multi_hop",
                "metric_type": "success_rate",
                "value": float(is_correct),
                "metadata": {
                    "num_hops": num_hops,
                    "statements": statements,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_answer": model_answer,
                    "confidence": confidence,
                    "explanation": explanation
                }
            })
            
        except Exception as e:
            logger.error(f"Error running multi-hop task: {str(e)}")
            raise
            
        return results 