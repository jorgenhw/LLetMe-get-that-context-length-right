import unittest
from src.data_structures import Document, Needle, EvaluationResult
from src.evaluation import EvaluationFramework
import pandas as pd
import numpy as np

class TestEvaluationFramework(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.framework = EvaluationFramework(self.api_key)
        
        # Create test document
        self.test_document = Document(
            id="test_doc",
            content="Test content",
            pages=["Page 1", "Page 2"],
            length_tokens=100,
            length_pages=2,
            needles=[]
        )
        
        # Create test needles
        self.test_needle = Needle(
            content="Test needle",
            page_number=1,
            position_percent=50.0,
            insertion_point="middle",
            query="What is the test needle?",
            expected_answer="Test needle",
            task_type="single"
        )
        
        # Add needle to document
        self.test_document.needles.append(self.test_needle)
        
        # Create test results
        self.test_result = EvaluationResult(
            model_name="test_model",
            document_id="test_doc",
            task_type="single",
            query="What is the test needle?",
            model_answer="Test needle",
            expected_answer="Test needle",
            score=2,
            is_correct=True,
            metadata={
                "page_number": 1,
                "position_percent": 50.0,
                "document_length": 100
            }
        )
        
        # Add result to framework
        self.framework.results.append(self.test_result)

    def test_calculate_metrics_overall_accuracy(self):
        metrics = self.framework.calculate_metrics()
        self.assertIn("overall_accuracy", metrics)
        self.assertEqual(metrics["overall_accuracy"], 1.0)

    def test_calculate_metrics_task_specific(self):
        metrics = self.framework.calculate_metrics()
        self.assertIn("single_accuracy", metrics)
        self.assertEqual(metrics["single_accuracy"], 1.0)
        self.assertIn("single_avg_score", metrics)
        self.assertEqual(metrics["single_avg_score"], 2.0)

    def test_calculate_metrics_position_bias(self):
        metrics = self.framework.calculate_metrics()
        self.assertIn("position_bias", metrics)
        self.assertIn("middle", metrics["position_bias"])
        self.assertEqual(metrics["position_bias"]["middle"], 1.0)

    def test_calculate_metrics_length_impact(self):
        metrics = self.framework.calculate_metrics()
        self.assertIn("length_impact", metrics)
        self.assertIn("medium", metrics["length_impact"])
        self.assertEqual(metrics["length_impact"]["medium"], 1.0)

    def test_save_results(self):
        import os
        test_file = "test_results.csv"
        self.framework.save_results(test_file)
        self.assertTrue(os.path.exists(test_file))
        os.remove(test_file)

    def test_empty_results(self):
        empty_framework = EvaluationFramework(self.api_key)
        metrics = empty_framework.calculate_metrics()
        self.assertEqual(metrics, {})

if __name__ == '__main__':
    unittest.main() 