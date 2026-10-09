# nanoagent

**A minimal AI agent in ~350 lines of Python. No frameworks, no dependencies, just the raw loop.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Dependencies 0](https://img.shields.io/badge/dependencies-0-brightgreen)

---

## Quick Start

```bash
git clone https://github.com/iamsnh/nanoagent.git
cd nanoagent

export OPENAI_API_KEY="sk-..."    # or ANTHROPIC_API_KEY
python example.py
```

That's it. No `pip install`, no virtual environment, no config files.

## What It Does

```python
from nanoagent import create_agent

agent, tools = create_agent()

@tools.register(
    name="calculator",
    description="Evaluate a math expression",
    parameters={"expression": {"type": "string", "description": "e.g. '2**10'"}},
)
def calculator(expression: str) -> str:
    return str(eval(expression))

answer = agent.run("What is 2^10 + 3^7?")
```

The agent will:
1. **Think** -- decide it needs to use the calculator
2. **Act** -- call `calculator(expression="2**10 + 3**7")`
3. **Observe** -- read the result `3211`
4. **Respond** -- "2^10 + 3^7 = 3,211"

## Architecture

nanoagent implements the [ReAct pattern](https://arxiv.org/abs/2210.03629) (Reason + Act) from scratch:

```
User Query
    |
    v
+-------------------+
|   ReAct Loop      |  (max N steps)
|                   |
|  1. LLM thinks    |----> "I need to search for X"
|  2. LLM calls     |----> tool_name(args)
|  3. Tool executes  |----> result string
|  4. Result feeds   |----> back into context
|     back to LLM   |
|                   |
|  Repeat until the  |
|  LLM responds     |
|  without a tool   |
|  call              |
+-------------------+
    |
    v
Final Answer
```

### Core Components

| Component | What it does | Lines |
|-----------|-------------|-------|
| `ToolRegistry` | Decorator-based tool registration + schema generation | ~60 |
| `Memory` | Sliding-window conversation history | ~40 |
| `OpenAIBackend` | Raw HTTP to OpenAI-compatible APIs (streaming) | ~80 |
| `AnthropicBackend` | Raw HTTP to Anthropic API (streaming) | ~70 |
| `Agent` | The ReAct loop with retry + backoff | ~60 |

### Key Design Decisions

- **Raw HTTP, no SDKs.** Uses `urllib` from the standard library. You can read every byte that goes over the wire.
- **Streaming by default.** Tokens print as they arrive. The streaming parser handles both SSE formats (OpenAI and Anthropic) correctly.
- **Sliding window memory.** Keeps the system prompt and the last N messages. No vector database, no embeddings -- just a window.
- **Automatic retry with exponential backoff.** Transient API errors (429, 500, network) are retried up to 3 times.
- **Backend auto-detection.** Set `ANTHROPIC_API_KEY` and it uses Claude. Set `OPENAI_API_KEY` and it uses GPT. Both? Anthropic wins.

## Comparison

| | nanoagent | LangChain | CrewAI | AutoGen |
|---|---|---|---|---|
| Lines of code | ~350 | ~600k | ~20k | ~50k |
| Dependencies | 0 | 100+ | 30+ | 40+ |
| Install time | 0s | 30s+ | 15s+ | 20s+ |
| Time to understand | 30 min | weeks | days | days |
| ReAct loop | yes | yes | yes | yes |
| Tool use | yes | yes | yes | yes |
| Streaming | yes | yes | partial | yes |
| Multi-provider | yes | yes | yes | yes |

## Why?

Because most agent frameworks are 1000x more code than they need to be.

The core of an AI agent is a loop: ask the LLM what to do, do it, feed the result back, repeat. That loop is ~50 lines. The rest is tool registration, API formatting, and retry logic. The entire thing fits in one file you can read in 30 minutes.

This is the same philosophy as Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [micrograd](https://github.com/karpathy/micrograd): strip away the frameworks and stare at the thing itself. If you understand nanoagent, you understand what LangChain, CrewAI, and every other agent framework actually do under the hood.

Use this to:
- **Learn** how AI agents really work
- **Prototype** an agent in 5 minutes
- **Customize** without fighting a framework
- **Teach** others the core concepts

## API

### `create_agent(**kwargs) -> (Agent, ToolRegistry)`

Creates an agent and its tool registry in one call.

```python
agent, tools = create_agent(
    max_steps=10,       # max ReAct iterations
    max_retries=3,      # retries per LLM call
    memory_window=40,   # messages to keep in context
    verbose=True,       # print reasoning to stderr
)
```

### `@tools.register(name, description, parameters)`

Register a function as a tool. The function must accept keyword arguments and return a string.

```python
@tools.register(
    name="search",
    description="Search the web",
    parameters={"query": {"type": "string", "description": "Search query"}},
)
def search(query: str) -> str:
    return do_search(query)
```

### `agent.run(user_input, stream=True) -> str`

Run the agent on a user query. Returns the final answer as a string.

## Using with Different Providers

```python
from nanoagent import Agent, ToolRegistry, OpenAIBackend, AnthropicBackend

# OpenAI
agent = Agent(backend=OpenAIBackend(model="gpt-4o"))

# Anthropic
agent = Agent(backend=AnthropicBackend(model="claude-sonnet-4-20250514"))

# Any OpenAI-compatible API (Ollama, Together, etc.)
agent = Agent(backend=OpenAIBackend(
    model="llama-3.1-70b",
    base_url="http://localhost:11434/v1",
))
```

## License

MIT. See [LICENSE](LICENSE).
