import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import agent


def text(t):
    return NS(type="text", text=t)


def tool(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


class FakeClient:
    """Plays back scripted model responses so the loop can be tested without an API."""
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw["messages"][-1])
        return self.script.pop(0)


class Tests(unittest.TestCase):
    def test_calculate(self):
        self.assertAlmostEqual(agent.calculate("0.17 * 2450"), 416.5)
        with self.assertRaises(ValueError):
            agent.calculate("__import__('os')")

    def test_tool_loop(self):
        c = FakeClient([NS(stop_reason="tool_use", content=[tool("1", "calculate", expression="2+2")]),
                        NS(stop_reason="end_turn", content=[text("The answer is 4")])])
        self.assertEqual(agent.run_agent("2+2?", c, verbose=False), "The answer is 4")
        self.assertEqual(c.calls[1]["content"][0]["content"], "4")

    def test_tool_error_is_returned_to_model(self):
        c = FakeClient([NS(stop_reason="tool_use", content=[tool("1", "read_file", path="../secret")]),
                        NS(stop_reason="end_turn", content=[text("done")])])
        agent.run_agent("x", c, verbose=False)
        self.assertTrue(c.calls[1]["content"][0]["is_error"])

    def test_step_limit(self):
        c = FakeClient([NS(stop_reason="tool_use", content=[tool(str(i), "calculate", expression="1")])
                        for i in range(3)])
        self.assertIn("step limit", agent.run_agent("x", c, max_steps=3, verbose=False))

    def test_workspace_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            agent.WORKSPACE = Path(d).resolve()
            (Path(d) / "a.txt").write_text("hello")
            self.assertEqual(agent.read_file("a.txt"), "hello")
            with self.assertRaises(PermissionError):
                agent.read_file("../etc/passwd")


if __name__ == "__main__":
    unittest.main()
