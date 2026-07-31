# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching repo demonstrating **DeepEval** evaluation across three shapes of LLM application:
a single-turn tool-calling agent, a RAG agent, and a multi-turn chatbot. Everything is
offline and deterministic (in-memory fake order/policy data) except the LLM calls themselves.

There is no application to deploy — the "product" is the pairing of each agent with its eval
scripts and the markdown explainers in `agent_evals/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+. Create `.env` (gitignored) from `.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-...        # the agents under test
OPENAI_API_KEY=sk-...               # GPT-4o judge, RAG embeddings, chatbot.py
CONFIDENT_API_KEY=confident_us_...  # optional: stream traces to Confident AI
```

## Running

Eval scripts are **standalone scripts, not pytest tests**. Run them with `python`, **from the
repo root** — each inserts the repo root into `sys.path` so `from agent_instrumented import ...`
resolves.

```bash
# Sanity-check an agent alone (no evals)
python agent_instrumented.py
python rag_agent.py
python chatbot.py                 # interactive; 'quit' to exit

# Single-turn agent evals
python agent_evals/test_TaskCompletion.py
python agent_evals/test_TracingComponentsTest.py
python agent_evals/test_multipleEvals.py
python agent_evals/test_CustomMetricEvals.py

# RAG
python agent_evals/rag/test_rag_agent.py

# Multi-turn
python agent_evals/multiturn/test_chatbot.py
python agent_evals/multiturn/test_custom_metrics_chatbot.py
```

## Project structure

### Agents (systems under test, repo root)

| File | Purpose |
|---|---|
| `agent_plain.py` | Baseline LangChain + Claude agent, two tools (`get_order_status`, `get_refund_policy`). No eval code — the "before" state shown to students. |
| `agent_instrumented.py` | Same agent + DeepEval instrumentation, plus one extra tool loaded over MCP from `mcp_server.py`. All `agent_evals/` scripts import `support_agent` from here. |
| `rag_agent.py` | RAG support agent: 9 policy docs in `InMemoryVectorStore` with OpenAI embeddings, one `search_policies` tool returning top-3 chunks. Exports `rag_support_agent`. |
| `chatbot.py` | Multi-turn chatbot on the raw OpenAI function-calling API (no LangChain). Owns its tool-call loop; `chat()` returns `(reply, history, tools_called)`. |
| `mcp_server.py` | FastMCP stdio server exposing `get_shipping_options`. Started as a subprocess by `agent_instrumented.py` at import time. |
| `policies.txt` | Human-readable policy reference. **Not imported by any code** — `rag_agent.py` has its own inline `POLICY_DOCS` list. |

### Evals (`agent_evals/`)

| Script | Metrics |
|---|---|
| `test_TaskCompletion.py` | `TaskCompletionMetric` — the minimal agentic eval, one golden. |
| `test_TracingComponentsTest.py` | `TaskCompletionMetric` + `ToolCorrectnessMetric`. Goldens carry `expected_tools`. |
| `test_multipleEvals.py` | `PromptAlignmentMetric`, `StepEfficiencyMetric`, `AnswerRelevancyMetric`. Input-only goldens. |
| `test_CustomMetricEvals.py` | `GEval` (custom "Correctness"). Goldens carry `expected_output`. |
| `rag/test_rag_agent.py` | `ContextualPrecisionMetric`, `ContextualRecallMetric`. Needs `expected_output` **and** `retrieval_context`. |
| `multiturn/test_chatbot.py` | `TurnRelevancyMetric`, `KnowledgeRetentionMetric`, `ConversationCompletenessMetric`. |
| `multiturn/test_custom_metrics_chatbot.py` | `ConversationalGEval` (custom "Correctness") over `MultiTurnParams`. |

## Architecture

**Model split.** Agents run on **Claude** (`claude-sonnet-4-6` via `ChatAnthropic`) —
`agent_plain.py`, `agent_instrumented.py`, `rag_agent.py`. `chatbot.py` runs on **GPT-4o**
via the OpenAI SDK directly. The judge for every LLM-based metric is **GPT-4o**, so the
Claude agents are never grading themselves. Keep it that way when adding metrics.

**Instrumentation pattern** (`agent_instrumented.py`, `rag_agent.py`):
1. `deepeval_callback = CallbackHandler()` — captures every LangChain LLM/tool span
2. `config={"callbacks": [deepeval_callback]}` on `.invoke()` / `.ainvoke()` — wires it in
3. `update_current_trace(output=reply)` — sets a clean final answer on the trace
4. The **eval script** adds `@observe(name="...")` on a thin wrapper around the imported agent

**Why the wrapper in the eval scripts.** DeepEval 4.0.4 reads `expected_tools`,
`expected_output`, and `retrieval_context` from the **trace object**, not from the `Golden`.
The `@observe`-decorated wrapper calls `get_current_golden()` and copies them across with
`update_current_trace(...)` before delegating to the real agent. This keeps the agent files
free of golden-specific code.

**Why `evals_iterator()` and not `evaluate()` for single-turn.** `TaskCompletionMetric` (and
the other trace-scoped metrics) need `test_case._trace_dict` populated, which only happens on
DeepEval's agentic path — `@observe` + `dataset.evals_iterator()`. Building a bare
`LLMTestCase` and calling `evaluate()` silently falls into a deprecated, broken fallback
template (a Jinja `tools_called_formatted is undefined` crash).

**RAG retrieval capture.** `search_policies` stashes the retrieved chunks in a module-level
`_last_retrieved` list; `rag_support_agent` forwards them via
`update_current_trace(retrieval_context=...)`. Without this, all Contextual/Faithfulness
metrics fail with no context.

**Multi-turn is a different model entirely.** No `Golden`, no trace. `test_chatbot.py` drives
`chat()` live to build a list of `Turn` objects, wraps them in a `ConversationalTestCase`, and
scores with `evaluate()`. Consequence: `ToolCorrectnessMetric` **cannot** be used here — tool
quality goes into a `ConversationalGEval` rubric (optionally reading
`MultiTurnParams.TOOLS_CALLED`).

**MCP integration.** `agent_instrumented.py` uses `MultiServerMCPClient` to spawn
`mcp_server.py` over stdio at import time and merges its tools into the agent's tool list.
This is why that file needs `nest_asyncio` and uses `agent.ainvoke` — the MCP client is
async and would otherwise clash with DeepEval's own event loop.

**Confident AI tracing.** Setting `CONFIDENT_API_KEY` in `.env` is the only change needed.
DeepEval reads it automatically; no code changes.

## Conventions

- Never add eval code to the agent files. Instrumentation (steps 1–3 above) is the only
  DeepEval that belongs there; anything golden-aware goes in the eval script's `@observe`
  wrapper.
- `agent_plain.py` must stay eval-free — its whole purpose is being the "before" half of the
  diff. Keep it in sync with `agent_instrumented.py` for everything except the instrumentation.
- New metric → add it to the metric cheat sheet and the eval-script table in `README.md`.
- Fake data (`ORDERS`, `REFUND_POLICIES`, `POLICY_DOCS`) is duplicated across agent files on
  purpose, so each file reads standalone in a lesson. Don't factor it into a shared module.

## Known issues

- **DeepEval 4.0.4 `_make_hashable` crash.** Unhashable `ToolMessage` objects in
  `tools_called` crash `ToolCorrectnessMetric`. The dev workaround was an in-place edit to
  `deepeval/test_case/llm_test_case.py` in site-packages (wrap `hash(obj)` in `try/except`,
  fall back to `str(obj)`). **This does not survive a fresh `pip install`** — if
  `test_TracingComponentsTest.py` crashes on an unhashable type, that's why.
- **`create_react_agent` deprecation.** Migrated to `from langchain.agents import create_agent`
  with `system_prompt=` replacing `prompt=`. Requires the base `langchain` package.
- **Version sensitivity.** Developed against `deepeval` 4.0.4, `langchain` 1.3.2,
  `langchain-core` 1.4.0, `langchain-anthropic` 1.4.3, `langchain-mcp-adapters` 0.2.2,
  `openai` 2.38.0. `requirements.txt` is unpinned; DeepEval's integration surface moves fast.