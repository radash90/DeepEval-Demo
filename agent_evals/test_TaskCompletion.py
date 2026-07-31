import os
import sys

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe

from agent_instrumented import support_agent as _support_agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?"),
])

for golden in dataset.evals_iterator(metrics=[TaskCompletionMetric(threshold=0.7, model="gpt-4o")]):
    support_agent(golden.input)
