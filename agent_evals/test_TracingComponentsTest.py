import os
import sys

from deepeval.contextvars import get_current_golden
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from agent_instrumented import support_agent as _support_agent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trace have expected and actual values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    golden = get_current_golden()
    if golden:
        if golden.expected_tools:
            update_current_trace(expected_tools=golden.expected_tools)
        if golden.expected_output:
            update_current_trace(expected_tools=golden.expected_output)

    return _support_agent(user_input)

taskCompletionMetric = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

tool_correctness = ToolCorrectnessMetric()

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?",
           expected_tools=[ToolCall(name="get_order_status")]),
    Golden(input="What if I don't like my shoes",
           expected_tools=[ToolCall(name="get_refund_policy")])
])

for golden in dataset.evals_iterator(metrics=[taskCompletionMetric, tool_correctness]):
    support_agent(golden.input)

