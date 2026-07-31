import os.path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import PromptAlignmentMetric, StepEfficiencyMetric, AnswerRelevancyMetric
from deepeval.tracing import observe
from agent_instrumented import support_agent as _support_agent


# Trace have expected and actual values
@observe(name="support_agent")
def support_agent(user_input: str) -> str:
    return _support_agent(user_input)


answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o");

## threshold of 0.5 is enough since we're more focused on the steps taken by the LLM/agent to provide the answer.
step_efficiency = StepEfficiencyMetric(threshold=0.5, model="gpt-4o")

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=[
        "You are a friendly customer-support agent. "
        "Keep replies short and helpful."
    ],
    threshold=0.7,
    model="gpt-4o"
)

dataset = EvaluationDataset(goldens=[
    Golden(input="Where is my order ORD-1042?"),
    Golden(input="What is the refund policy on electronics?"),
    Golden(input="I want to return order ORD-2099, what should I do?")
])

for golden in dataset.evals_iterator(metrics=[prompt_alignment, step_efficiency, answer_relevancy_metric]):
    support_agent(golden.input)
