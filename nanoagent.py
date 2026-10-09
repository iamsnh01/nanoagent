"""
nanoagent — A minimal AI agent in ~350 lines. No frameworks, no dependencies, just the raw loop.
Implements ReAct (Reason + Act) from scratch using only the Python standard library.
Works with OpenAI and Anthropic APIs via raw HTTP requests.
"""
from __future__ import annotations
import json, os, sys, time, urllib.request, urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

# --- Tool registry -----------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any]) -> Callable:
        """Decorator to register a function as a tool the agent can call."""
        def decorator(fn: Callable[..., str]) -> Callable[..., str]:
            self._tools[name] = Tool(name=name, description=description, parameters=parameters, fn=fn)
            return fn
        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return str(tool.fn(**arguments))
        except Exception as e:
            return f"Error executing {name}: {e}"

    def schema_for_openai(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": t.name, "description": t.description,
            "parameters": {"type": "object", "properties": t.parameters,
                           "required": list(t.parameters.keys())},
        }} for t in self._tools.values()]

    def schema_for_anthropic(self) -> list[dict]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": {"type": "object", "properties": t.parameters,
                                  "required": list(t.parameters.keys())},
        } for t in self._tools.values()]

    def description_block(self) -> str:
        lines = []
        for t in self._tools.values():
            params = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines)

# --- Conversation memory with sliding window ---------------------------------

@dataclass
class Message:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None

class Memory:
    def __init__(self, max_messages: int = 40) -> None:
        self.messages: list[Message] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append(Message(role=role, content=content, **kwargs))
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) <= self.max_messages:
            return
        system = [m for m in self.messages if m.role == "system"]
        rest = [m for m in self.messages if m.role != "system"]
        keep = self.max_messages - len(system)
        self.messages = system + rest[-keep:]

    def to_openai(self) -> list[dict]:
        out = []
        for m in self.messages:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.name:
                d["name"] = m.name
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            out.append(d)
        return out

    def to_anthropic(self) -> tuple[str, list[dict]]:
        system = ""
        msgs: list[dict] = []
        for m in self.messages:
            if m.role == "system":
                system = m.content
            elif m.role == "tool":
                msgs.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.content}]})
            else:
                msgs.append({"role": m.role, "content": m.content})
        return system, msgs

# --- HTTP helpers (raw urllib, no dependencies) ------------------------------

def _http_post(url: str, headers: dict[str, str], body: dict) -> bytes:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.read()

def _http_stream(url: str, headers: dict[str, str], body: dict) -> Generator[bytes, None, None]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req)
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            yield line
    finally:
        resp.close()

# --- LLM backends ------------------------------------------------------------

@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = "end"

class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o", base_url: str = "https://api.openai.com/v1") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def chat(self, memory: Memory, tools: ToolRegistry, stream: bool = False) -> LLMResponse | Generator[str, None, LLMResponse]:
        body: dict[str, Any] = {"model": self.model, "messages": memory.to_openai()}
        schemas = tools.schema_for_openai()
        if schemas:
            body["tools"] = schemas
        return self._stream(body) if stream else self._complete(body)

    def _complete(self, body: dict) -> LLMResponse:
        raw = _http_post(f"{self.base_url}/chat/completions", self._headers(), body)
        data = json.loads(raw)
        choice = data["choices"][0]
        msg = choice["message"]
        tool_calls = []
        for tc in msg.get("tool_calls", []):
            tool_calls.append({"id": tc["id"], "name": tc["function"]["name"],
                               "arguments": json.loads(tc["function"]["arguments"])})
        return LLMResponse(content=msg.get("content") or "", tool_calls=tool_calls,
                           stop_reason=choice.get("finish_reason", "stop"))

    def _stream(self, body: dict) -> Generator[str, None, LLMResponse]:
        body["stream"] = True
        chunks: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        for line in _http_stream(f"{self.base_url}/chat/completions", self._headers(), body):
            text = line.decode().strip()
            if not text.startswith("data: ") or text == "data: [DONE]":
                continue
            delta = json.loads(text[6:]).get("choices", [{}])[0].get("delta", {})
            if "content" in delta and delta["content"]:
                chunks.append(delta["content"])
                yield delta["content"]
            for tc in delta.get("tool_calls", []):
                idx = tc["index"]
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": tc.get("id", ""), "name": "", "arguments": ""}
                if "id" in tc and tc["id"]:
                    tool_calls_acc[idx]["id"] = tc["id"]
                if "function" in tc:
                    tool_calls_acc[idx]["name"] += tc["function"].get("name", "")
                    tool_calls_acc[idx]["arguments"] += tc["function"].get("arguments", "")
        parsed = []
        for tc in tool_calls_acc.values():
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            parsed.append({"id": tc["id"], "name": tc["name"], "arguments": args})
        return LLMResponse(content="".join(chunks), tool_calls=parsed, stop_reason="stop")

class AnthropicBackend:
    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"}

    def chat(self, memory: Memory, tools: ToolRegistry, stream: bool = False) -> LLMResponse | Generator[str, None, LLMResponse]:
        system, messages = memory.to_anthropic()
        body: dict[str, Any] = {"model": self.model, "max_tokens": 4096, "messages": messages}
        if system:
            body["system"] = system
        schemas = tools.schema_for_anthropic()
        if schemas:
            body["tools"] = schemas
        return self._stream(body) if stream else self._complete(body)

    def _complete(self, body: dict) -> LLMResponse:
        raw = _http_post("https://api.anthropic.com/v1/messages", self._headers(), body)
        data = json.loads(raw)
        text_parts, tool_calls = [], []
        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({"id": block["id"], "name": block["name"], "arguments": block["input"]})
        return LLMResponse(content="".join(text_parts), tool_calls=tool_calls,
                           stop_reason=data.get("stop_reason", "end"))

    def _stream(self, body: dict) -> Generator[str, None, LLMResponse]:
        body["stream"] = True
        chunks: list[str] = []
        tool_calls: list[dict] = []
        current_tool: dict | None = None
        input_json = ""
        for line in _http_stream("https://api.anthropic.com/v1/messages", self._headers(), body):
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            event = json.loads(text[6:])
            etype = event.get("type", "")
            if etype == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool = {"id": block["id"], "name": block["name"], "arguments": {}}
                    input_json = ""
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunks.append(delta["text"])
                    yield delta["text"]
                elif delta.get("type") == "input_json_delta":
                    input_json += delta.get("partial_json", "")
            elif etype == "content_block_stop" and current_tool is not None:
                try:
                    current_tool["arguments"] = json.loads(input_json) if input_json else {}
                except json.JSONDecodeError:
                    current_tool["arguments"] = {}
                tool_calls.append(current_tool)
                current_tool = None
                input_json = ""
        return LLMResponse(content="".join(chunks), tool_calls=tool_calls, stop_reason="end_turn")

# --- The Agent (the ReAct loop) ----------------------------------------------

REACT_SYSTEM = """You are a helpful agent that solves problems step-by-step using the ReAct pattern.

For each step:
1. Thought: reason about what to do next
2. Action: call a tool if needed
3. Observation: read the tool result

Available tools:
{tools}

When you have the final answer, respond directly without calling a tool."""

class Agent:
    def __init__(
        self,
        backend: OpenAIBackend | AnthropicBackend | None = None,
        tools: ToolRegistry | None = None,
        max_steps: int = 10,
        max_retries: int = 3,
        memory_window: int = 40,
        verbose: bool = True,
    ) -> None:
        self.backend = backend or self._auto_backend()
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.memory = Memory(max_messages=memory_window)
        self.verbose = verbose

    @staticmethod
    def _auto_backend() -> OpenAIBackend | AnthropicBackend:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicBackend()
        return OpenAIBackend()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def run(self, user_input: str, stream: bool = True) -> str:
        system_prompt = REACT_SYSTEM.format(tools=self.tools.description_block())
        if not self.memory.messages or self.memory.messages[0].role != "system":
            self.memory.add("system", system_prompt)
        self.memory.add("user", user_input)

        for step in range(self.max_steps):
            self._log(f"\n--- Step {step + 1}/{self.max_steps} ---")
            response = self._call_llm(stream=stream)
            if not response.tool_calls:
                self._log("[Agent] Final answer reached.")
                self.memory.add("assistant", response.content)
                return response.content
            self.memory.add("assistant", response.content)
            for tc in response.tool_calls:
                self._log(f"[Tool] {tc['name']}({json.dumps(tc['arguments'])})")
                result = self.tools.execute(tc["name"], tc["arguments"])
                self._log(f"[Result] {result[:200]}{'...' if len(result) > 200 else ''}")
                self.memory.add("tool", result, name=tc["name"], tool_call_id=tc["id"])

        self._log("[Agent] Max steps reached.")
        return response.content if response else "I could not complete the task within the step limit."

    def _call_llm(self, stream: bool) -> LLMResponse:
        for attempt in range(self.max_retries):
            try:
                result = self.backend.chat(self.memory, self.tools, stream=stream)
                if stream and not isinstance(result, LLMResponse):
                    return self._consume_stream(result)
                return result
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
                wait = 2 ** attempt
                self._log(f"[Retry] Attempt {attempt + 1}/{self.max_retries} failed: {e}. Waiting {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries")

    def _consume_stream(self, gen: Generator[str, None, LLMResponse]) -> LLMResponse:
        try:
            while True:
                token = next(gen)
                if self.verbose:
                    print(token, end="", flush=True)
        except StopIteration as e:
            if self.verbose:
                print()
            return e.value

# --- Convenience: create agent + registry in one shot ------------------------

def create_agent(**kwargs: Any) -> tuple[Agent, ToolRegistry]:
    tools = ToolRegistry()
    agent = Agent(tools=tools, **kwargs)
    return agent, tools
