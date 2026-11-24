"""Graphe LangGraph de la phase 2 : un agent (Backend Developer) + validation humaine obligatoire.

    START → prepare → developer → human_review ─┬─ approve → merge   → END
                         ▲                      ├─ changes → developer (avec ton retour)
                         └──────────────────────┘
                                                └─ reject  → discard → END

human_review utilise interrupt() : le programme s'arrête et l'état est sauvegardé (SQLite).
Il reprend seulement quand tu donnes ta décision (python -m squad decide ...).
"""
import json
import subprocess
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import gitops, prompts
from .agent import run_agent
from .config import SQUAD_ROOT
from .tracing import Tracer
from .workspace import TOOL_SPECS, Workspace

DECISIONS = {"approve", "changes", "reject"}


class SquadState(TypedDict, total=False):
    run_id: str
    target: str
    task: dict
    branch: str
    iteration: int
    feedback: str
    summary: str
    files: list[str]
    commits: list[str]
    decision: str
    result: str


def human_name() -> str:
    r = subprocess.run(["git", "config", "user.name"], cwd=SQUAD_ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "Project Owner"


def build_graph(chat_factory: Callable, checkpointer, quiet: bool = False):
    def tracer(s: SquadState) -> Tracer:
        return Tracer(s["target"], s["run_id"], quiet=quiet)

    def prepare(s: SquadState) -> dict:
        t, task = tracer(s), s["task"]
        gitops.check_ready(s["target"])
        branch = f"squad/{task['id']}-{s['run_id'][-6:]}"
        gitops.create_branch(s["target"], branch)
        t.record("task", "orchestrator", {"task": task, "branch": branch})
        t.event("orchestrator", f"tâche {task['id']} « {task['title']} » → branche {branch} → {task['role']}")
        return {"branch": branch, "iteration": 0, "commits": []}

    def developer(s: SquadState) -> dict:
        t, task, agent = tracer(s), s["task"], s["task"]["role"]
        it = s.get("iteration", 0) + 1
        tpl = prompts.load(SQUAD_ROOT / task["prompt_template"])
        p = prompts.render(
            tpl,
            task_id=task["id"], task_title=task["title"], task_description=task["description"].strip(),
            acceptance="\n".join(f"- {a}" for a in task.get("acceptance", [])),
            write_allow=", ".join(task["write_allow"]),
        )
        user = p.user
        if s.get("feedback"):
            user += f"\n\n## Retour du Project Owner (itération {it})\n{s['feedback']}\nCorrige en conséquence."
        t.record("prompt", agent, {"iteration": it, "template": tpl.path, "template_id": p.template_id,
                                    "template_version": p.template_version, "system": p.system, "user": user})
        t.event(agent, f"itération {it} : démarrage (prompt {p.template_id} v{p.template_version})")

        ws = Workspace(s["target"], task["write_allow"])
        chat = chat_factory(p.system, TOOL_SPECS, t, agent)
        res = run_agent(chat, ws, user, t, agent)
        t.write_text(f"conversation_{agent}_iter{it}.json",
                     json.dumps(chat.transcript(), ensure_ascii=False, indent=2, default=str))
        t.record("agent_result", agent, {"iteration": it, "summary": res.summary, "steps": res.steps,
                                          "stop_reason": res.stop_reason, "files_written": ws.written})

        sha = ""
        if ws.written:
            msg = (f"{task['id']} : {task['title']} (itération {it})\n\n{res.summary[:1500]}\n\n"
                   f"Agent: {agent}\nTask: {task['id']}\nRun-Id: {s['run_id']}\nIteration: {it}")
            sha = gitops.commit_paths(s["target"], ws.written, msg, identity=gitops.agent_identity(agent))
        t.event("orchestrator", f"commit {sha or '(aucun fichier modifié)'} sur {s['branch']}")
        return {"iteration": it, "summary": res.summary, "files": ws.written,
                "commits": s.get("commits", []) + ([sha] if sha else []), "feedback": ""}

    def human_review(s: SquadState) -> dict:
        t, it = tracer(s), s["iteration"]
        patch = t.write_text(f"review_iter{it}.patch", gitops.diff(s["target"], s["branch"]))
        answer = interrupt({
            "run_id": s["run_id"], "task": s["task"]["id"], "title": s["task"]["title"],
            "iteration": it, "branch": s["branch"], "summary": s.get("summary", ""),
            "diff_stat": gitops.diff_stat(s["target"], s["branch"]), "patch": str(patch),
        })
        decision = answer.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"décision invalide : {decision!r} (attendu : {sorted(DECISIONS)})")
        t.record("decision", "human", {"iteration": it, "decision": decision,
                                        "feedback": answer.get("feedback", ""), "decided_by": human_name()})
        t.event("human", f"décision : {decision}" + (f" — « {answer.get('feedback')} »" if answer.get("feedback") else ""))
        return {"decision": decision, "feedback": answer.get("feedback", "")}

    def merge(s: SquadState) -> dict:
        t = tracer(s)
        sha = gitops.merge(s["target"], s["branch"],
                           f"Merge {s['branch']} : {s['task']['id']} approuvé par {human_name()}\n\nRun-Id: {s['run_id']}")
        t.event("orchestrator", f"fusion dans main ({sha})")
        t.record("result", "orchestrator", {"result": "merged", "merge_commit": sha})
        gitops.commit_paths(s["target"], [f"ai-trace/runs/{s['run_id']}"], f"Trace {s['run_id']} ({s['task']['id']}, approuvé)")
        return {"result": "merged"}

    def discard(s: SquadState) -> dict:
        t = tracer(s)
        gitops.discard(s["target"], s["branch"])
        t.event("orchestrator", f"branche {s['branch']} supprimée (travail rejeté, conservé dans la trace)")
        t.record("result", "orchestrator", {"result": "rejected"})
        gitops.commit_paths(s["target"], [f"ai-trace/runs/{s['run_id']}"], f"Trace {s['run_id']} ({s['task']['id']}, rejeté)")
        return {"result": "rejected"}

    g = StateGraph(SquadState)
    for name, fn in [("prepare", prepare), ("developer", developer), ("human_review", human_review),
                     ("merge", merge), ("discard", discard)]:
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "developer")
    g.add_edge("developer", "human_review")
    g.add_conditional_edges("human_review", lambda s: {"approve": "merge", "changes": "developer",
                                                        "reject": "discard"}[s["decision"]],
                            ["merge", "developer", "discard"])
    g.add_edge("merge", END)
    g.add_edge("discard", END)
    return g.compile(checkpointer=checkpointer)
