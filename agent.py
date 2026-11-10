"""A small tool-using AI agent (ReAct-style loop) built on the Anthropic Messages API.

The model decides which tool to call, the loop runs it, feeds the result back, and repeats
until the model answers or the step limit is hit.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python agent.py "How many .py files are in the workspace, and what is 17% of 2450?"
"""
import ast
import operator
import os
import sys
from pathlib import Path

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", ".")).resolve()
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
SYSTEM = ("You are a careful assistant with tools. Use tools for arithmetic and for anything "
          "about files; never guess. When you have the answer, reply concisely.")

# ---------- tools ----------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg}


def calculate(expression):
    """Safely evaluate an arithmetic expression (no eval)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expression, mode="eval").body)


def _safe_path(rel):
    p = (WORKSPACE / rel).resolve()
    if WORKSPACE != p and WORKSPACE not in p.parents:
        raise PermissionError("path is outside the workspace")
    return p


def list_files(directory="."):
    """List files in a workspace directory."""
    return "\n".join(sorted(x.name for x in _safe_path(directory).iterdir()))


def read_file(path, max_chars=4000):
    """Read the start of a text file in the workspace."""
    return _safe_path(path).read_text(errors="replace")[:max_chars]


TOOLS = {"calculate": calculate, "list_files": list_files, "read_file": read_file}
SCHEMAS = [
    {"name": "calculate", "description": "Evaluate an arithmetic expression, e.g. '0.17 * 2450'.",
     "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
    {"name": "list_files", "description": "List files in a workspace directory.",
     "input_schema": {"type": "object", "properties": {"directory": {"type": "string"}}}},
    {"name": "read_file", "description": "Read a text file from the workspace.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                      "required": ["path"]}},
]


# ---------- agent loop ----------
def run_agent(task, client, model=MODEL, max_steps=8, verbose=True):
    messages = [{"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        resp = client.messages.create(model=model, max_tokens=1024, system=SYSTEM,
                                      tools=SCHEMAS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                out, err = TOOLS[block.name](**block.input), False
            except Exception as e:  # report tool errors to the model so it can recover
                out, err = f"Error: {e}", True
            if verbose:
                print(f"[step {step}] {block.name}({block.input}) -> {str(out)[:80]}")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": str(out), "is_error": err})
        messages.append({"role": "user", "content": results})
    return "Stopped: step limit reached before the agent finished."


if __name__ == "__main__":
    import anthropic
    print(run_agent(" ".join(sys.argv[1:]) or "What is 17% of 2450?", anthropic.Anthropic()))
