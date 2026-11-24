"""Opérations Git, réservées à l'orchestrateur (les agents n'ont aucun accès à Git).

- Chaque tâche travaille sur sa branche squad/<id>.
- Les commits des agents sont signés au nom de l'agent, avec Agent/Task/Run-Id dans le message.
- Seule une approbation humaine fusionne dans main.
"""
import subprocess
from pathlib import Path

MAIN = "main"


class GitError(Exception):
    pass


def git(repo: str | Path, *args: str, identity: tuple[str, str] | None = None) -> str:
    cmd = ["git"]
    if identity:
        cmd += ["-c", f"user.name={identity[0]}", "-c", f"user.email={identity[1]}"]
    r = subprocess.run(cmd + list(args), cwd=repo, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)} → {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def agent_identity(agent: str) -> tuple[str, str]:
    return f"Squad Agent ({agent})", f"{agent.replace('_', '-')}@squad.local"


def check_ready(repo) -> None:
    if git(repo, "rev-parse", "--abbrev-ref", "HEAD") != MAIN:
        raise GitError(f"le dépôt cible doit être sur la branche {MAIN}")
    dirty = [l for l in git(repo, "status", "--porcelain").splitlines() if "ai-trace/" not in l]
    if dirty:
        raise GitError(f"le dépôt cible a des modifications non commitées : {dirty}")


def create_branch(repo, branch: str) -> None:
    git(repo, "checkout", "-b", branch, MAIN)


def commit_paths(repo, paths: list[str], message: str, identity=None) -> str:
    git(repo, "add", "--", *paths)
    if not git(repo, "diff", "--cached", "--name-only"):
        return ""
    git(repo, "commit", "-m", message, identity=identity)
    return git(repo, "rev-parse", "--short", "HEAD")


def changed_paths(repo, allow: list[str]) -> list[str]:
    """Fichiers modifiés ou nouveaux (non commités) situés dans les dossiers autorisés de la tâche.

    Sert après une reprise (squad resume) : les écritures d'une itération interrompue restent dans
    l'arbre de travail et doivent être commitées avec l'itération reprise (bug vu le 24/11/2026 sur T001).
    """
    prefixes = [a.strip("/") + "/" for a in allow]
    out = []
    for line in git(repo, "status", "--porcelain", "-uall").splitlines():
        # Format « XY chemin » ; git() retire l'espace initial de la 1re ligne (« M x » au lieu de «  M x ») :
        # on découpe donc sur le premier blanc au lieu d'une position fixe.
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        path = parts[1].strip().strip('"')
        if any(path.startswith(p) for p in prefixes):
            out.append(path)
    return out


def diff(repo, branch: str) -> str:
    return git(repo, "diff", f"{MAIN}...{branch}")


def diff_stat(repo, branch: str) -> str:
    return git(repo, "diff", "--stat", f"{MAIN}...{branch}")


def merge(repo, branch: str, message: str) -> str:
    git(repo, "checkout", MAIN)
    git(repo, "merge", "--no-ff", branch, "-m", message)
    return git(repo, "rev-parse", "--short", "HEAD")


def discard(repo, branch: str) -> None:
    git(repo, "checkout", MAIN)
    git(repo, "branch", "-D", branch)
