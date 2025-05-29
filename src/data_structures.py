from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Needle:
    content: str
    page_number: int
    position_percent: float
    insertion_point: str
    query: str
    expected_answer: str
    task_type: str  # 'single_needle', 'multi_needle', 'multi_hop', 'aggregation'
    hop_count: Optional[int] = None  # For multi-hop tasks
    chain_elements: Optional[List[str]] = None  # For multi-hop tasks
    components: Optional[List[str]] = None  # For aggregation tasks
    num_needles: Optional[int] = None  # For multi-needle tasks
    related_needles: Optional[List[str]] = None  # For multi-needle tasks

@dataclass
class Document:
    id: str
    content: str
    pages: List[str]
    length_tokens: int
    length_pages: int
    needles: List[Needle]

@dataclass
class EvaluationResult:
    model_name: str
    document_id: str
    task_type: str
    query: str
    model_answer: str
    expected_answer: str
    score: int  # 0, 1, or 2
    is_correct: bool
    metadata: Dict  # Enhanced metadata for specific task types
