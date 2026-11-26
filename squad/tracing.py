"""Traçabilité : chaque prompt, réponse brute, appel d'outil et décision humaine devient un fichier JSON.

Emplacement : <dépôt cible>/ai-trace/runs/<run_id>/NNN_<agent>_<type>.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

console = Console()

AGENT_COLORS = {"orchestrator": "cyan", "backend_developer": "green", "frontend_developer": "blue", "reviewer": "yellow", "human": "magenta"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Tracer:
    def __init__(self, target: str | Path, run_id: str, quiet: bool = False):
        self.run_id = run_id
        self.dir = Path(target) / "ai-trace" / "runs" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet

    def record(self, kind: str, agent: str, data: dict) -> Path:
        seq = len(list(self.dir.glob("[0-9][0-9][0-9]_*.json"))) + 1
        path = self.dir / f"{seq:03d}_{agent}_{kind}.json"
        payload = {"run_id": self.run_id, "seq": seq, "agent": agent, "kind": kind, "timestamp": now(), **data}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def event(self, agent: str, message: str) -> None:
        """Affichage en direct dans le terminal + journal texte."""
        with open(self.dir / "events.log", "a", encoding="utf-8") as f:
            f.write(f"{now()} [{agent}] {message}\n")
        if not self.quiet:
            color = AGENT_COLORS.get(agent, "white")
            console.print(f"  [{color}][{agent}][/{color}] {message}", highlight=False)
