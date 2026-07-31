import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.tracing import update_current_trace, observe
from rag_agent import rag_support_agent as _rag_support_agent

# Trace have expected and actual values
@observe(name="rag_support_agent")
def rag_support_agent(user_input: str) -> str:
    golden = get_current_golden()
    update_current_trace(expected_output=golden.expected_output)
    return _rag_support_agent(user_input)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4o"
)

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4o"
)

dataset = EvaluationDataset(goldens=[
    Golden(
        input="What is the return policy for electronics",
           expected_output="Electronics can be returned within 15 fays of delivery if unopened"
                            "and in original packaging. Refunds take about 5-7 business days."
        ),

    Golden(input="How long does express shipping take?",
           expected_output="Express shipping takes 1-2 business days and costs $15. "
           "Order placed before 2pm are dispatched the same day."
           )
])

for golden in dataset.evals_iterator(metrics=[contextual_precision_metric, contextual_recall_metric]):
    rag_support_agent(golden.input)