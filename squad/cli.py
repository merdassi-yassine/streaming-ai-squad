"""Ligne de commande de supervision.

  python -m squad run <tache.yaml> --target <dépôt>   lance une tâche, s'arrête à ta validation
  python -m squad show <run_id>                        réaffiche le résumé et le diff en attente
  python -m squad decide <run_id> approve|changes|reject [--feedback "..."]
  python -m squad graph                                affiche le graphe (Mermaid, pour le rapport)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .config import STATE_DB, get_settings
from .graph import build_graph
from .llm import make_chat

console = Console()


def chat_factory():
    settings = get_settings()
    return lambda system, tools, tracer, agent: make_chat(settings, system, tools, tracer, agent)


def show_pending(result_or_state, with_diff: bool = True) -> None:
    interrupts = result_or_state.get("__interrupt__") if isinstance(result_or_state, dict) else None
    if not interrupts:
        return
    p = interrupts[0].value
    tests = p.get("tests")
    if not tests:
        tests_txt = "non configurés pour cette tâche"
    else:
        status = "[green]✓ OK[/green]" if tests["passed"] else f"[red]✗ ÉCHEC (code {tests['exit_code']})[/red]"
        tests_txt = f"{status} — exécutés par l'orchestrateur ({tests['log']})\n{tests['tail']}"
    console.print(Panel.fit(
        f"[bold]{p['task']} — {p['title']}[/bold]  (itération {p['iteration']}, branche {p['branch']})\n\n"
        f"[bold]Résumé de l'agent développeur :[/bold]\n{p['summary'] or '(vide)'}\n\n"
        f"[bold]Fichiers :[/bold]\n{p['diff_stat'] or '(aucun changement)'}\n\n"
        f"[bold]Tests réels :[/bold] {tests_txt}"
        + (f"\n\n[bold]Avis de l'agent relecteur :[/bold]\n{p['review']}" if p.get("review") else ""),
        title="⏸ VALIDATION HUMAINE REQUISE", border_style="magenta"))
    if with_diff:
        diff = Path(p["patch"]).read_text(encoding="utf-8")
        if diff:
            console.print(Syntax(diff, "diff", theme="ansi_dark", word_wrap=True))
    console.print(f"Trace complète : {Path(p['patch']).parent}")
    console.print(
        f"\n[bold]Ta décision :[/bold]\n"
        f"  python -m squad decide {p['run_id']} approve\n"
        f"  python -m squad decide {p['run_id']} changes --feedback \"ce qu'il faut corriger\"\n"
        f"  python -m squad decide {p['run_id']} reject")


def report_end(result: dict) -> None:
    if result.get("result"):
        console.print(f"\n[bold green]Terminé : {result['result']}[/bold green]")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="squad")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("task"); r.add_argument("--target", required=True)
    s = sub.add_parser("show"); s.add_argument("run_id")
    d = sub.add_parser("decide"); d.add_argument("run_id"); d.add_argument("decision", choices=["approve", "changes", "reject"])
    d.add_argument("--feedback", default="")
    sub.add_parser("graph")
    a = ap.parse_args(argv)

    if a.cmd == "graph":
        print(build_graph(lambda *x: None, InMemorySaver()).get_graph().draw_mermaid())
        return 0

    STATE_DB.parent.mkdir(exist_ok=True)
    with SqliteSaver.from_conn_string(str(STATE_DB)) as cp:
        if a.cmd == "run":
            task = yaml.safe_load(Path(a.task).read_text(encoding="utf-8"))
            run_id = f"{task['id']}-{datetime.now():%Y%m%d-%H%M%S}"
            graph = build_graph(chat_factory(), cp)
            console.print(f"[bold]▶ Run {run_id}[/bold]")
            result = graph.invoke({"run_id": run_id, "target": str(Path(a.target).resolve()), "task": task},
                                  {"configurable": {"thread_id": run_id}})
        else:
            cfg = {"configurable": {"thread_id": a.run_id}}
            graph = build_graph(chat_factory() if a.cmd == "decide" else (lambda *x: None), cp)
            snap = graph.get_state(cfg)
            if not snap.interrupts:
                console.print(f"[red]Aucune validation en attente pour {a.run_id}[/red]")
                return 1
            if a.cmd == "show":
                show_pending({"__interrupt__": snap.interrupts})
                return 0
            if a.decision == "changes" and not a.feedback:
                console.print("[red]--feedback est obligatoire avec 'changes'[/red]")
                return 1
            result = graph.invoke(Command(resume={"decision": a.decision, "feedback": a.feedback}), cfg)
        show_pending(result)
        report_end(result)
    return 0
