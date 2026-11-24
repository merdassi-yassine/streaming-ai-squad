"""Boucle d'un agent : le LLM demande des outils, on les exécute (contrôlés), on lui renvoie les résultats."""
from dataclasses import dataclass

from .tracing import Tracer
from .workspace import Workspace

MAX_STEPS = 15  # garde-fou : un agent qui tourne en rond s'arrête


@dataclass
class AgentResult:
    summary: str
    steps: int
    stop_reason: str  # "done" | "max_steps"


def _short(args: dict) -> str:
    if "content" in args:
        return f"{args.get('path')} ({len(args['content'].splitlines())} lignes)"
    return ", ".join(f"{v}" for v in args.values()) or "."


def run_agent(chat, workspace: Workspace, task_text: str, tracer: Tracer, agent: str) -> AgentResult:
    turn = chat.send_user(task_text)
    for step in range(1, MAX_STEPS + 1):
        if not turn.tool_calls:
            return AgentResult(turn.text.strip(), step, "done")
        results = []
        for call in turn.tool_calls:
            res = workspace.execute(call.name, call.args)
            status = "✓" if res.get("ok") else ("⛔ REFUSÉ" if res.get("refused") else "✗")
            tracer.event(agent, f"{call.name} {_short(call.args)} {status}")
            tracer.record("tool_call", agent, {"tool": call.name, "args": call.args, "result": res})
            results.append((call.name, res))
        turn = chat.send_tool_results(results)
    return AgentResult(turn.text.strip(), MAX_STEPS, "max_steps")
