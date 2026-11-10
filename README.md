# ai-agent

A small tool-using AI agent in Python. The model decides which tool to call, the program runs it and feeds the result back, and the loop repeats until the model gives a final answer.

## How it works

This is a ReAct-style loop built on the Anthropic Messages API:

1. Send the task and the tool definitions to the model.
2. If the model asks to use a tool, run it and send the result back.
3. Repeat until the model answers or the step limit is reached.

## Tools

| Tool | What it does |
| --- | --- |
| `calculate` | Evaluates arithmetic safely by parsing the expression, with no `eval` |
| `list_files` | Lists files in a workspace folder |
| `read_file` | Reads the start of a text file in the workspace |

## Design choices

- **Sandboxed file access:** paths are resolved and checked, so the agent cannot read files outside the workspace folder.
- **Tool errors go back to the model:** a failed tool call returns an error message instead of crashing, so the agent can recover and try again.
- **Step limit:** the loop stops after a fixed number of steps so it can never run forever.
- **Testable without an API:** the client is passed in, so the tests use a scripted fake model.

## Setup

Requires Python 3.9 or newer and an Anthropic API key.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
python agent.py "What is 17% of 2450, and how many files are in this folder?"
```

Optional settings: `AGENT_MODEL` changes the model and `AGENT_WORKSPACE` changes the folder the file tools can access.

## Tests

```bash
python -m unittest test_agent -v
```

The tests cover safe arithmetic, the tool loop, error handling, the step limit and the workspace sandbox. They need no API key.

## Ideas for extending it

- Add a web search or HTTP tool
- Keep memory between runs
- Add a planning step before tool use
