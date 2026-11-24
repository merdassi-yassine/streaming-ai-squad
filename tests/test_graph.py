"""Test de bout en bout du graphe avec un FAUX LLM (aucun appel réseau, résultat déterministe)."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from squad.graph import build_graph
from squad.llm import ToolCall, Turn

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@local")
    (tmp_path / "README.md").write_text("sandbox\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


class FakeChat:
    """Rejoue un scénario : écrit un fichier (+ une tentative interdite), puis rend un résumé."""
    instances = []

    def __init__(self, version):
        self.version = version
        self.sent = []
        FakeChat.instances.append(self)

    def send_user(self, text):
        self.sent.append(text)
        return Turn("", [ToolCall("write_file", {"path": "hello/greeting.py", "content": f"V = {self.version}\n"}),
                         ToolCall("write_file", {"path": "../evil.py", "content": "x"})], "fake-model")

    def send_tool_results(self, results):
        self.sent.append(results)
        return Turn(f"Résumé v{self.version}", [], "fake-model")

    def transcript(self):
        return [str(s) for s in self.sent]


def make_graph():
    FakeChat.instances = []
    factory = lambda system, tools, tracer, agent: FakeChat(len(FakeChat.instances) + 1)
    return build_graph(factory, InMemorySaver(), quiet=True)


def task():
    return yaml.safe_load((ROOT / "tasks" / "T000-hello.yaml").read_text(encoding="utf-8"))


def trace_kinds(repo, run_id):
    return [json.loads(p.read_text(encoding="utf-8"))["kind"]
            for p in sorted((repo / "ai-trace" / "runs" / run_id).glob("[0-9]*.json"))]


def test_approbation_fusionne_dans_main(repo):
    g, cfg = make_graph(), {"configurable": {"thread_id": "r1"}}
    out = g.invoke({"run_id": "T000-r1", "target": str(repo), "task": task()}, cfg)

    # 1) Le graphe s'arrête AVANT toute intégration : main ne contient pas encore le fichier.
    assert out["__interrupt__"][0].value["iteration"] == 1
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD").startswith("squad/T000")
    assert "Agent: backend_developer" in git(repo, "log", "-1", "--format=%B")
    assert git(repo, "log", "-1", "--format=%an") == "Squad Agent (backend_developer)"
    assert git(repo, "show", "main:README.md") == "sandbox"  # main intact
    with pytest.raises(subprocess.CalledProcessError):
        git(repo, "show", "main:hello/greeting.py")

    # 2) Approbation humaine → fusion + trace commitée sur main.
    out = g.invoke(Command(resume={"decision": "approve"}), cfg)
    assert out["result"] == "merged"
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(repo, "show", "main:hello/greeting.py") == "V = 1"
    assert not (repo.parent / "evil.py").exists()
    kinds = trace_kinds(repo, "T000-r1")
    assert kinds[:2] == ["task", "prompt"]
    assert {"tool_call", "agent_result", "decision", "result"} <= set(kinds)
    refused = [json.loads(p.read_text(encoding="utf-8")) for p in (repo / "ai-trace/runs/T000-r1").glob("*tool_call.json")]
    assert any(r["result"].get("refused") for r in refused)
    assert git(repo, "status", "--porcelain") == ""  # tout est commité


def test_corrections_puis_rejet(repo):
    g, cfg = make_graph(), {"configurable": {"thread_id": "r2"}}
    g.invoke({"run_id": "T000-r2", "target": str(repo), "task": task()}, cfg)

    out = g.invoke(Command(resume={"decision": "changes", "feedback": "Ajoute une docstring"}), cfg)
    assert out["__interrupt__"][0].value["iteration"] == 2
    assert "Ajoute une docstring" in FakeChat.instances[1].sent[0]  # ton retour est transmis à l'agent

    out = g.invoke(Command(resume={"decision": "reject"}), cfg)
    assert out["result"] == "rejected"
    assert "squad/" not in git(repo, "branch")  # branche supprimée
    with pytest.raises(subprocess.CalledProcessError):
        git(repo, "show", "main:hello/greeting.py")  # rien n'a été intégré
    run = repo / "ai-trace/runs/T000-r2"
    assert (run / "review_iter1.patch").exists() and (run / "review_iter2.patch").exists()
    assert trace_kinds(repo, "T000-r2").count("decision") == 2
    assert "Trace T000-r2" in git(repo, "log", "-1", "--format=%s")


def test_tests_reels_et_relecteur(repo):
    """L'orchestrateur lance VRAIMENT la commande de test ; le relecteur ne peut rien écrire."""
    import sys
    g, cfg = make_graph(), {"configurable": {"thread_id": "r4"}}
    t = task() | {"test_command": [sys.executable, "-c", "print('3 passed'); raise SystemExit(1)"], "review": True}
    out = g.invoke({"run_id": "T000-r4", "target": str(repo), "task": t}, cfg)

    pending = out["__interrupt__"][0].value
    assert pending["tests"]["exit_code"] == 1 and pending["tests"]["passed"] is False  # vrai code de retour
    assert "3 passed" in pending["tests"]["tail"]
    assert pending["review"] == "Résumé v2"  # 2e agent (relecteur) appelé après le développeur
    run = repo / "ai-trace/runs/T000-r4"
    assert (run / "tests_iter1.log").read_text(encoding="utf-8").startswith("3 passed")
    assert {"test_run", "review"} <= set(trace_kinds(repo, "T000-r4"))
    # Le relecteur a tenté d'écrire (faux LLM) : refusé, rien de commité en son nom.
    assert "reviewer" not in git(repo, "log", "--all", "--format=%an")
    assert git(repo, "show", f"{pending['branch']}:hello/greeting.py") == "V = 1"


def test_decision_invalide_refusee(repo):
    g, cfg = make_graph(), {"configurable": {"thread_id": "r3"}}
    g.invoke({"run_id": "T000-r3", "target": str(repo), "task": task()}, cfg)
    with pytest.raises(ValueError, match="décision invalide"):
        g.invoke(Command(resume={"decision": "yolo"}), cfg)
