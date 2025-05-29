import unittest
from unittest.mock import Mock, patch
from src.data_structures import Document, Needle
import os

class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        # Do not create self.pipeline here!
        # Create a valid mock Document
        self.mock_doc = Document(
            id="test_doc",
            content="Test content",
            pages=["Page 1"],
            length_tokens=100,
            length_pages=1,
            needles=[]
        )
        self.test_pdf = "test.pdf"
        with open(self.test_pdf, "wb") as f:
            f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")

    def tearDown(self):
        if os.path.exists(self.test_pdf):
            os.remove(self.test_pdf)
        for file in os.listdir():
            if file.startswith("results_") and file.endswith(".csv"):
                os.remove(file)

    @patch('src.pipeline.DocumentProcessor')
    @patch('src.pipeline.NeedleGenerator')
    def test_prepare_documents(self, mock_needle_generator, mock_doc_processor):
        from src.pipeline import EvaluationPipeline
        mock_doc_processor.return_value.create_document.return_value = self.mock_doc
        pipeline = EvaluationPipeline(self.api_key)
        documents = pipeline.prepare_documents([self.test_pdf])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].id, "test_doc")

    @patch('openai.OpenAI')
    @patch('src.pipeline.NeedleGenerator')
    @patch('src.pipeline.MultiHopGenerator')
    @patch('src.pipeline.AggregationGenerator')
    def test_inject_needles(self, mock_agg_generator, mock_multihop_generator, mock_needle_generator, mock_openai):
        from src.pipeline import EvaluationPipeline
        pipeline = EvaluationPipeline(self.api_key)
        doc = Document(
            id="test_doc",
            content="Test content",
            pages=["Page 1"],
            length_tokens=100,
            length_pages=1,
            needles=[]
        )
        mock_needle = Needle(
            content="Test needle",
            page_number=1,
            position_percent=50.0,
            insertion_point="middle",
            query="What is the test needle?",
            expected_answer="Test needle",
            task_type="single"
        )
        mock_needle_generator.return_value.generate_context_aware_needle.return_value = mock_needle
        # Patch insert_needle_into_document to actually add the needle to the document
        def insert_needle_side_effect(document, needle):
            document.needles.append(needle)
            return document
        mock_needle_generator.return_value.insert_needle_into_document.side_effect = insert_needle_side_effect
        mock_openai.return_value.chat.completions.create.return_value.choices = [
            Mock(message=Mock(content='{"needle": "Test needle", "query": "What is the test needle?", "expected_answer": "Test needle", "insertion_point": "middle"}'))
        ]
        documents = pipeline.inject_needles([doc])
        self.assertEqual(len(documents), 1)
        self.assertTrue(len(documents[0].needles) > 0)

    @patch('src.pipeline.DocumentProcessor')
    @patch('src.pipeline.NeedleGenerator')
    @patch('src.pipeline.MultiHopGenerator')
    @patch('src.pipeline.AggregationGenerator')
    @patch('src.pipeline.EvaluationFramework')
    def test_run_complete_evaluation(self, mock_eval_framework, mock_agg_generator, mock_multihop_generator, mock_needle_generator, mock_doc_processor):
        from src.pipeline import EvaluationPipeline
        mock_doc_processor.return_value.create_document.return_value = self.mock_doc
        pipeline = EvaluationPipeline(self.api_key)
        mock_model = Mock()
        mock_model.query_model.return_value = "Test answer"
        self.mock_doc.needles = [
            Needle(
                content="Test needle",
                page_number=1,
                position_percent=50.0,
                insertion_point="middle",
                query="What is the test needle?",
                expected_answer="Test needle",
                task_type="single"
            )
        ]
        mock_needle = self.mock_doc.needles[0]
        mock_needle_generator.return_value.generate_context_aware_needle.return_value = mock_needle
        def insert_needle_side_effect(document, needle):
            document.needles.append(needle)
            return document
        mock_needle_generator.return_value.insert_needle_into_document.side_effect = insert_needle_side_effect
        # Patch multi-hop and aggregation generators to do nothing
        mock_multihop_generator.return_value.generate_multi_hop_chain.side_effect = lambda doc, num_hops: doc
        mock_agg_generator.return_value.generate_aggregation_task.side_effect = lambda doc, num_items: doc
        mock_eval_framework.return_value.calculate_metrics.return_value = {
            "overall_accuracy": 1.0,
            "single_accuracy": 1.0
        }
        results = pipeline.run_complete_evaluation([self.test_pdf], {"test_model": mock_model})
        self.assertIn("test_model", results)
        self.assertEqual(results["test_model"]["overall_accuracy"], 1.0)

if __name__ == '__main__':
    unittest.main() 