import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepeval.evaluate import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric, ConversationCompletenessMetric
from deepeval.test_case import ConversationalTestCase, Turn

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

turn_relevancy_metric = TurnRelevancyMetric(threshold=0.7)
knowledge_retention_metric = KnowledgeRetentionMetric(threshold=0.5)
conversation_completeness_metric = ConversationCompletenessMetric(threshold=0.5)

test_case = ConversationalTestCase(
    turns=turns
)

evaluate(test_cases=[test_case], metrics=[turn_relevancy_metric,
                                          knowledge_retention_metric, conversation_completeness_metric])
