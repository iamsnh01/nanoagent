"""
example.py — Demo of nanoagent solving a multi-step research question.

Run:  OPENAI_API_KEY=sk-... python example.py
  or: ANTHROPIC_API_KEY=sk-... python example.py
"""

from nanoagent import create_agent

agent, tools = create_agent(max_steps=8, verbose=True)

# --- Tool definitions -------------------------------------------------------

@tools.register(
    name="calculator",
    description="Evaluate a mathematical expression. Returns the numeric result.",
    parameters={"expression": {"type": "string", "description": "A Python math expression, e.g. '2 ** 10 + 3 * 7'"}},
)
def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/.() %eE")
    if not all(c in allowed for c in expression):
        return f"Error: unsafe expression '{expression}'"
    try:
        return str(eval(expression))  # safe: only numeric chars allowed
    except Exception as e:
        return f"Error: {e}"


@tools.register(
    name="web_search",
    description="Search the web for a query. Returns a summary of top results.",
    parameters={"query": {"type": "string", "description": "The search query"}},
)
def web_search(query: str) -> str:
    # Mock search results for demo purposes.
    # In production, replace with a real search API call.
    results = {
        "population of france": "France has a population of approximately 68.4 million (2024 estimate).",
        "population of germany": "Germany has a population of approximately 84.5 million (2024 estimate).",
        "area of france": "France covers approximately 640,679 square kilometers.",
        "gdp of france": "France's GDP is approximately $3.05 trillion USD (2024).",
    }
    query_lower = query.lower()
    for key, value in results.items():
        if key in query_lower:
            return value
    return f"Search results for '{query}': No specific data found. Try a more specific query."


@tools.register(
    name="read_file",
    description="Read the contents of a local file.",
    parameters={"path": {"type": "string", "description": "Path to the file to read"}},
)
def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            content = f.read(10_000)
        return content if content else "(empty file)"
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except Exception as e:
        return f"Error reading file: {e}"

# --- Run the agent -----------------------------------------------------------

if __name__ == "__main__":
    question = (
        "What is the population density of France? "
        "First find the population, then the area in square km, "
        "then calculate the density."
    )
    print(f"Question: {question}\n")
    answer = agent.run(question)
    print(f"\nFinal answer: {answer}")
