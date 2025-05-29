from typing import Dict, List, Tuple
import openai
import json
import random
from pathlib import Path
import logging
from ..data_structures import Document, Needle

logger = logging.getLogger(__name__)

class AggregationTask:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_statements(self, context: str, document_style: str) -> Tuple[List[str], str, str]:
        """Generate related but independent statements and a synthesis question"""
        
        # Get a summary of the document by taking the first 1000 words
        context_summary = ' '.join(context.split()[:1000])
        
        prompt = f"""
You are tasked with creating related but independent statements for document evaluation.

Context from document (first 1000 words):
{context_summary}

Document style/type: {document_style}

Requirements:
1. Create 2-3 related but independent statements that:
   - Are realistic for legal contracts
   - Each contains specific, verifiable information
   - Are independent of each other
   - Could plausibly be part of this document
   - Together can answer a higher-level question
2. Create a synthesis question that:
   - Requires combining ALL statements to answer
   - Has a straightforward combination (sum, list, etc.)
   - Is specific and unambiguous
3. Provide the exact answer that requires combining all statements

Respond in JSON format:
{{
    "statements": [
        "Statement 1",
        "Statement 2",
        "Statement 3"  # optional
    ],
    "question": "Question that requires combining all statements",
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
            logger.error(f"Error generating statements: {str(e)}")
            raise

    def insert_statements(self, document: Document) -> Tuple[Document, List[str], str, str]:
        """Insert statements into different sections of the document"""
        
        # Get total number of pages
        num_pages = len(document.pages)
        num_statements = random.randint(2, 3)
        
        if num_pages < num_statements:
            raise ValueError(f"Document has fewer pages ({num_pages}) than required statements ({num_statements})")
        
        # Generate the statements
        statements, question, answer = self.generate_statements(
            context=document.content,
            document_style="formal document"
        )
        
        # Randomly select pages for each statement
        available_pages = list(range(num_pages))
        random.shuffle(available_pages)
        selected_pages = available_pages[:len(statements)]
        
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
        """Evaluate if the model's answer correctly synthesizes all information"""
        
        eval_prompt = f"""
Evaluate if the model's answer correctly synthesizes all required information.

Expected Answer: {expected_answer}
Model Answer: {model_answer}

Requirements:
- All required information must be correctly combined
- Minor wording differences are acceptable
- The synthesis must be accurate

Respond in JSON format:
{{
    "is_correct": true/false,
    "completeness": 0.0-1.0,  # |Required ∩ Found| / |Required|
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
            return result["is_correct"], result["completeness"], result["explanation"]
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            raise

    def run(self, document: Document, model_inference) -> List[Dict]:
        """Run the aggregation task on a document"""
        
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
            
            # Insert statements and get question
            modified_doc, statements, question, expected_answer = self.insert_statements(doc_copy)
            
            # Get model's answer
            model_answer = model_inference.query_model(modified_doc, question)
            
            # Evaluate answer
            is_correct, completeness, explanation = self.evaluate_response(
                model_answer=model_answer,
                expected_answer=expected_answer
            )
            
            # Record results
            results.append({
                "model_name": model_inference.model_name,
                "document_id": document.id,  # Use the original document ID
                "task_type": "aggregation",
                "metric_type": "completeness",
                "value": completeness,
                "metadata": {
                    "statements": statements,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "explanation": explanation
                }
            })
            
        except Exception as e:
            logger.error(f"Error running aggregation task: {str(e)}")
            raise
            
        return results 