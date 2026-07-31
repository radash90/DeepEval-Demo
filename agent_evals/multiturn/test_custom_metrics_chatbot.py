import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric, GEval, \
    ConversationalGEval
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams

from chatbot import chat

questions_list = ["Hi, I placed an order last week. The order ID is ORD-1042 ",
                  "Will it arrive on time?", "What was the ETA you just mentioned",
                  "Can I upgrade to express shipping?"]

turns = []
history = []

for user_msg in questions_list:
    reply, history, _ = chat(user_msg, history)
    turns.append(Turn(role="user", content=user_msg))
    turns.append(Turn(role="assistant", content=reply))

conversational_gEval_metric = ConversationalGEval(
    name="Correctness",
    criteria=(
        "Did the chatbot fully resolve the customer issue?"
        "It should use tools when needed and provide accurate answers."
    ),
    model="gpt-4o",
    threshold=0.7,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT,
    ]
)

test_case = ConversationalTestCase(
    turns=turns
)

evaluate(test_cases=[test_case], metrics=[conversational_gEval_metric])
