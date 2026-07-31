import os.path
import sys

from deepeval.test_case import SingleTurnParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric
from deepeval.metrics import GEval
from deepeval.tracing import observe
from deepeval.dataset import get_current_golden
from deepeval.tracing.context import update_current_trace
from agent_instrumented import support_agent as _support_agent


# Trace have expected and actual values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    # GEval reads expected_output from the TRACE, not the Golden, so copy it in.
    golden = get_current_golden()
    update_current_trace(expected_output=golden.expected_output)
    return _support_agent(user_input)


geval_metric = GEval(
    name="Correctness",
    criteria=(
        "Determine whether the actual output conveys the same factual information "
        "as the expected output. Minor wording differences are acceptable; "
        "missing or wrong facts are not."
    ),
    model="gpt-4o",
    threshold=0.8,
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.ACTUAL_OUTPUT
    ]
)

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?",
           expected_output="The order ORD-1042 has been shipped and is expected to arrive on 14th May 2026"),

    Golden(input="What is the refund policy on electronics?",
           expected_output="Yes they can be returned within 15 days if they are unopened"),
    Golden(input="I want to return order ORD-2099, what should I do?",
           expected_output="Order ORD-2099 was delivered. Can you tell me the product category to pull up the refund policy?")
])

for golden in dataset.evals_iterator(metrics=[geval_metric]):
    support_agent(golden.input)
