# DeepEval Agentic Testing Demo

A hands-on repo for **evaluating LLM agents with [DeepEval](https://deepeval.com/docs/introduction)**.

It walks through three shapes of agentic evaluation, each with a runnable agent and a set
of eval scripts:

1. **Single-turn tool-calling agent** — Task Completion, Tool Correctness, Prompt Alignment,
   Step Efficiency, Answer Relevancy, custom GEval
2. **RAG agent** — Contextual Precision / Recall over a retrieved policy knowledge base
3. **Multi-turn chatbot** — Turn Relevancy, Knowledge Retention, Conversation Completeness,
   Conversational GEval

**Cross-vendor by design:** the agents run on **Claude** (`claude-sonnet-4-6`) while the judge
for every LLM-based metric is **GPT-4o**, so the model under test is never grading itself.
(The multi-turn chatbot is the exception — it runs on GPT-4o directly, to show the same eval
patterns against a non-LangChain, raw function-calling app.)

---

## Quick start

```bash
# 1. Create a virtualenv and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set your API keys
cp .env.example .env      # then fill in the real values

# 3. Sanity-check an agent on its own (no evals)
python agent_instrumented.py

# 4. Run an eval
python agent_evals/test_TaskCompletion.py
```

Requires **Python 3.11+**.

### API keys

`.env` (gitignored) needs:

| Variable | Required? | Used for |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | The agents under test (`agent_plain.py`, `agent_instrumented.py`, `rag_agent.py`) |
| `OPENAI_API_KEY` | Yes | The GPT-4o judge for all LLM-based metrics, RAG embeddings, and `chatbot.py` |
| `CONFIDENT_API_KEY` | Optional | Streams traces + eval results to the [Confident AI](https://app.confident-ai.com) dashboard. DeepEval picks it up automatically — no code change needed. |

---

## The teaching point

The diff between `agent_plain.py` and `agent_instrumented.py` is essentially **4 lines**.
That's the entire ask evaluation makes of application code:

1. `from deepeval.integrations.langchain import CallbackHandler`
2. `deepeval_callback = CallbackHandler()`
3. `config={"callbacks": [deepeval_callback]}` on `.invoke()`
4. `update_current_trace(output=reply)` so the trace carries a clean final answer

Everything else — the tools, the prompt, the model — stays exactly the same. The eval scripts
add a thin `@observe`-decorated wrapper on top and never modify the agent.

Show both files side by side. Once the diff looks tiny, component-level evaluation stops
being intimidating.

---

## Repo layout

### Agents (the systems under test)

| File | What it is |
|---|---|
| `agent_plain.py` | Baseline LangChain + Claude support agent with two tools (`get_order_status`, `get_refund_policy`). **Zero eval code** — the "before" state. |
| `agent_instrumented.py` | The same agent plus DeepEval instrumentation, and one extra tool loaded over **MCP** from `mcp_server.py`. Every eval under `agent_evals/` imports from here. |
| `rag_agent.py` | Retrieval-augmented support agent: 9 policy documents in an `InMemoryVectorStore` (OpenAI embeddings), one `search_policies` tool. Forwards retrieved chunks to the trace as `retrieval_context`. |
| `chatbot.py` | Multi-turn chatbot on the raw OpenAI function-calling API (no LangChain). Owns its tool-call loop and returns `(reply, history, tools_called)` per turn. Run it directly for an interactive session. |
| `mcp_server.py` | Tiny FastMCP stdio server exposing `get_shipping_options`. Consumed by `agent_instrumented.py` via `langchain-mcp-adapters`. |
| `policies.txt` | Plain-text copy of the support policies, kept as human reference. Not imported by any code — `rag_agent.py` holds its own inline `POLICY_DOCS`. |

### Evals (`agent_evals/`)

All eval scripts are **standalone scripts, not pytest tests** — run them with `python`, from the
repo root.

| Script | Metrics | Notes |
|---|---|---|
| `test_TaskCompletion.py` | `TaskCompletionMetric` | Smallest possible agentic eval: one golden, one metric. Start here. |
| `test_TracingComponentsTest.py` | `TaskCompletionMetric`, `ToolCorrectnessMetric` | Goldens carry `expected_tools`; the wrapper copies them onto the trace with `update_current_trace`. |
| `test_multipleEvals.py` | `PromptAlignmentMetric`, `StepEfficiencyMetric`, `AnswerRelevancyMetric` | Three independent axes on one run. Goldens are input-only — no reference answer needed. |
| `test_CustomMetricEvals.py` | `GEval` (custom "Correctness") | Goldens carry `expected_output`; the wrapper copies it onto the trace. |
| `rag/test_rag_agent.py` | `ContextualPrecisionMetric`, `ContextualRecallMetric` | Scores **retrieval quality**, not just the answer. Needs both `expected_output` and `retrieval_context`. |
| `multiturn/test_chatbot.py` | `TurnRelevancyMetric`, `KnowledgeRetentionMetric`, `ConversationCompletenessMetric` | Drives `chatbot.py` live over 4 turns, then scores the conversation. |
| `multiturn/test_custom_metrics_chatbot.py` | `ConversationalGEval` (custom "Correctness") | Custom rubric applied across the whole conversation. |

```bash
python agent_evals/test_TaskCompletion.py
python agent_evals/test_TracingComponentsTest.py
python agent_evals/test_multipleEvals.py
python agent_evals/test_CustomMetricEvals.py
python agent_evals/rag/test_rag_agent.py
python agent_evals/multiturn/test_chatbot.py
python agent_evals/multiturn/test_custom_metrics_chatbot.py
```

> Run these from the **repo root**. The scripts add the repo root to `sys.path` so
> `from agent_instrumented import ...` resolves.

---

## The three evaluation shapes

Understanding these three shapes is most of what this repo teaches:

| | Single-turn agent | RAG | Multi-turn |
|---|---|---|---|
| Test case | `Golden` | `Golden` | `ConversationalTestCase` + `Turn`s |
| Runner | `dataset.evals_iterator()` | `dataset.evals_iterator()` | `evaluate()` |
| What's scored | One `@observe` trace | Same trace + `retrieval_context` | A list of turns (no trace, no golden) |

A consequence worth knowing: `ToolCorrectnessMetric` **cannot** run on a
`ConversationalTestCase`. In multi-turn, "did it use the right tools" is folded into a
`ConversationalGEval` rubric instead.

---

## Metric cheat sheet

| Metric | LLM judge? | Needs `expected_output`? | Needs `expected_tools`? | Needs `retrieval_context`? |
|---|---|---|---|---|
| `TaskCompletionMetric` | Yes | No | No | No |
| `ToolCorrectnessMetric` | No (name match) | No | **Yes** | No |
| `PromptAlignmentMetric` | Yes | No | No | No |
| `StepEfficiencyMetric` | Yes | No | No | No |
| `AnswerRelevancyMetric` | Yes | No | No | No |
| `GEval` (Correctness) | Yes | **Yes** | No | No |
| `ContextualPrecisionMetric` | Yes | **Yes** | No | **Yes** |
| `ContextualRecallMetric` | Yes | **Yes** | No | **Yes** |
| `TurnRelevancyMetric` | Yes | No | No | No |
| `KnowledgeRetentionMetric` | Yes | No | No | No |
| `ConversationCompletenessMetric` | Yes | No | No | No |
| `ConversationalGEval` | Yes | No | No | No |

---

## Suggested lesson flow

| Minute | Activity |
|---|---|
| 0–5 | Show `agent_plain.py`. Ask: "How would you test this?" |
| 5–10 | Show `agent_instrumented.py` side by side. Walk the 4-line diff. |
| 10–20 | Walk `agent_evals/test_TaskCompletion.py` line by line, then run it live. |
| 20–35 | `test_TracingComponentsTest.py` — introduce `expected_tools` and the deterministic vs. LLM-judged distinction. |
| 35–50 | `test_multipleEvals.py` and `test_CustomMetricEvals.py` — stacking metrics, writing your own rubric. |
| 50–70 | `rag/` — why retrieval needs its own metrics. |
| 70–90 | `multiturn/` — why conversations break the golden/trace model entirely. |

---

## Known issues

- **DeepEval 4.0.4 `_make_hashable` crash.** `ToolMessage` objects land in `tools_called` and
  aren't hashable, which crashes `ToolCorrectnessMetric`. The workaround used during
  development was an in-place edit to
  `deepeval/test_case/llm_test_case.py` in site-packages (wrap `hash(obj)` in a `try/except`
  and fall back to `str(obj)`). **This does not survive a fresh `pip install`** — if
  `test_TracingComponentsTest.py` crashes on an unhashable type, that's the cause.
- **Version sensitivity.** DeepEval's integration surface moves quickly. Pinned-ish versions
  this repo was developed against: `deepeval` 4.0.4, `langchain` 1.3.2, `langchain-core` 1.4.0,
  `langchain-anthropic` 1.4.3, `openai` 2.38.0. If an import path or a `Golden` field errors,
  check the DeepEval changelog before assuming the code is wrong.
- **Model availability.** The agents pin `claude-sonnet-4-6`. If that errors, swap in the
  current model id from the Anthropic console.

---

## License

MIT