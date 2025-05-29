import unittest
from unittest.mock import patch, Mock
import os
import sys
from pathlib import Path
import io

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from main import main

class TestMain(unittest.TestCase):
    def setUp(self):
        # Create test PDF file
        self.test_pdf = "test.pdf"
        with open(self.test_pdf, "wb") as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
        
        # Create test .env file
        self.env_file = ".env"
        with open(self.env_file, "w") as f:
            f.write('OPENAI_API_KEY="test-key"')

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_pdf):
            os.remove(self.test_pdf)
        if os.path.exists(self.env_file):
            os.remove(self.env_file)
        for file in os.listdir():
            if file.startswith("results_") and file.endswith(".csv"):
                os.remove(file)

    @patch('main.EvaluationPipeline')
    @patch('main.OpenAIInference')
    def test_main_with_env_file(self, mock_openai_inference, mock_pipeline):
        # Mock the pipeline
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.run_complete_evaluation.return_value = {
            "gpt-4": {
                "overall_accuracy": 1.0,
                "single_accuracy": 1.0,
                "multi_accuracy": 1.0,
                "hop_accuracy": 1.0,
                "aggregation_accuracy": 1.0
            }
        }

        # Mock OpenAI inference
        mock_openai_instance = Mock()
        mock_openai_inference.return_value = mock_openai_instance

        # Test main function
        with patch('sys.argv', ['main.py', '--pdf_files', self.test_pdf]):
            main()

        # Verify pipeline was called
        mock_pipeline_instance.run_complete_evaluation.assert_called_once()

    @patch('main.EvaluationPipeline')
    @patch('main.OpenAIInference')
    def test_main_with_multiple_pdfs(self, mock_openai_inference, mock_pipeline):
        # Create second test PDF
        test_pdf2 = "test2.pdf"
        with open(test_pdf2, "wb") as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

        try:
            # Mock the pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.run_complete_evaluation.return_value = {
                "gpt-4": {
                    "overall_accuracy": 1.0,
                    "single_accuracy": 1.0,
                    "multi_accuracy": 1.0,
                    "hop_accuracy": 1.0,
                    "aggregation_accuracy": 1.0
                }
            }

            # Mock OpenAI inference
            mock_openai_instance = Mock()
            mock_openai_inference.return_value = mock_openai_instance

            # Test main function with multiple PDFs
            with patch('sys.argv', ['main.py', '--pdf_files', self.test_pdf, test_pdf2]):
                main()

            # Verify pipeline was called
            mock_pipeline_instance.run_complete_evaluation.assert_called_once()
            self.assertEqual(len(mock_pipeline_instance.run_complete_evaluation.call_args[0][0]), 2)

        finally:
            # Clean up second test PDF
            if os.path.exists(test_pdf2):
                os.remove(test_pdf2)

if __name__ == '__main__':
    unittest.main() 