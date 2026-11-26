"""Rapport de traçabilité : résume tous les runs archivés dans <dépôt>/ai-trace/runs/ en un fichier Markdown.

Pour chaque run : tâche, prompts (modèle + version), appels au LLM (modèles réellement utilisés, erreurs,
jetons), fichiers écrits, résultats RÉELS des tests, avis du relecteur, décisions humaines, résultat final.
Tout est relu depuis les fichiers JSON de trace : le rapport n'invente rien.
"""
import json
import re
from collections import Counter
from pathlib import Path


def _load(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted(run_dir.glob("[0-9][0-9][0-9]_*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def _first_line(text: str, n: int = 160) -> str:
    line = next((l.strip() for l in (text or "").splitlines() if l.strip()), "")
    return line[:n] + ("…" if len(line) > n else "")


def summarize_run(run_dir: Path) -> dict:
    recs = _load(run_dir)
    task = next((r["task"] for r in recs if r["kind"] == "task"), {})
    calls = [r for r in recs if r["kind"] == "llm_call"]
    models = Counter(r.get("model_used") for r in calls)
    errors = Counter(str(e.get("code")) for r in calls for e in r.get("errors_before_success", []))
    tokens_in = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in calls)
    tokens_out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in calls)
    prompts = sorted({f"{r.get('template_id')} v{r.get('template_version')}" for r in recs if r["kind"] == "prompt"})
    tests = [r for r in recs if r["kind"] == "test_run"]
    reviews = [r for r in recs if r["kind"] == "review"]
    decisions = [r for r in recs if r["kind"] == "decision"]
    results = [r for r in recs if r["kind"] == "result"]
    refused = [r for r in recs if r["kind"] == "tool_call" and (r.get("result") or {}).get("refused")]
    written = sorted({f for r in recs if r["kind"] == "agent_result" for f in r.get("files_written", [])})
    iterations = max([r.get("iteration", 0) for r in recs if r["kind"] == "agent_result"] or [0])
    interrupted = (run_dir / "INTERRUPTED.txt").exists()
    cancelled = (run_dir / "CANCELLED.txt").exists()
    return {
        "run_id": run_dir.name, "task": task, "iterations": iterations, "prompts": prompts,
        "llm_calls": len(calls), "models": dict(models), "errors": dict(errors),
        "tokens_in": tokens_in, "tokens_out": tokens_out, "files": written, "refused": len(refused),
        "tests": [(t["iteration"], t["exit_code"], _last_summary(t.get("tail", ""))) for t in tests],
        "reviews": [(r["iteration"], _recommendation(r.get("review", ""))) for r in reviews],
        "decisions": [(d["iteration"], d["decision"], d.get("feedback", ""), d.get("decided_by", "")) for d in decisions],
        "result": results[-1]["result"] if results else ("annulé" if cancelled else "interrompu" if interrupted
                                                          else "en attente de décision"),
    }


def _last_summary(tail: str) -> str:
    m = re.findall(r"=+ (.*?(?:passed|failed|error).*?) =+", tail or "")
    return m[-1] if m else _first_line(tail, 80)


def _recommendation(review: str) -> str:
    m = re.search(r"Recommandation\s*:\s*\**\s*(APPROUVER|DEMANDER DES CORRECTIONS|REJETER)", review or "", re.I)
    return m.group(1).upper() if m else _first_line(review, 80)


def render(summaries: list[dict], target: str) -> str:
    total_calls = sum(s["llm_calls"] for s in summaries)
    lines = [
        "# Rapport de traçabilité de l'équipe d'agents IA",
        "",
        f"Dépôt analysé : `{target}` — {len(summaries)} run(s), {total_calls} appels au LLM.",
        "Généré automatiquement à partir des traces JSON (`ai-trace/runs/`) par `python -m squad report`.",
        "Les résultats de tests proviennent de l'exécution réelle par l'orchestrateur, jamais du LLM.",
        "",
        "## Vue d'ensemble",
        "",
        "| Run | Tâche | Itérations | Tests (dernier) | Relecteur (dernier) | Résultat |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        last_test = s["tests"][-1] if s["tests"] else None
        test_txt = "—" if not last_test else ("✓ " if last_test[1] == 0 else "✗ ") + last_test[2]
        review_txt = s["reviews"][-1][1] if s["reviews"] else "—"
        lines.append(f"| `{s['run_id']}` | {s['task'].get('id', '?')} {s['task'].get('title', '')} | "
                     f"{s['iterations']} | {test_txt} | {review_txt} | **{s['result']}** |")
    for s in summaries:
        t = s["task"]
        lines += ["", f"## {t.get('id', '?')} — {t.get('title', '')} (`{s['run_id']}`)", "",
                  f"- **Rôle / prompts :** {t.get('role', '?')} — {', '.join(s['prompts']) or '—'}",
                  f"- **Dossiers autorisés en écriture :** {', '.join(t.get('write_allow', []))}",
                  f"- **Appels au LLM :** {s['llm_calls']} — modèles réellement utilisés : "
                  + (", ".join(f"`{m}` ×{n}" for m, n in s["models"].items()) or "—"),
                  f"- **Erreurs d'API rencontrées (relancées) :** "
                  + (", ".join(f"{c} ×{n}" for c, n in sorted(s["errors"].items())) or "aucune"),
                  f"- **Jetons :** {s['tokens_in']:,} en entrée / {s['tokens_out']:,} en sortie".replace(",", " "),
                  f"- **Fichiers écrits par l'agent :** {', '.join(f'`{f}`' for f in s['files']) or '—'}",
                  f"- **Actions refusées par les garde-fous :** {s['refused']}",
                  "", "| Itération | Tests réels | Avis du relecteur | Décision humaine |", "|---|---|---|---|"]
        its = sorted({i for i, *_ in s["tests"]} | {i for i, _ in s["reviews"]} | {i for i, *_ in s["decisions"]})
        for i in its:
            tst = next((("✓ " if c == 0 else f"✗ (code {c}) ") + txt for j, c, txt in s["tests"] if j == i), "—")
            rev = next((r for j, r in s["reviews"] if j == i), "—")
            dec = next((f"**{d}**" + (f" — « {fb[:140]} »" if fb else "") + (f" ({who})" if who else "")
                        for j, d, fb, who in s["decisions"] if j == i), "—")
            lines.append(f"| {i} | {tst} | {rev} | {dec} |")
    return "\n".join(lines) + "\n"


def build_report(target: str | Path) -> str:
    runs = Path(target) / "ai-trace" / "runs"
    dirs = sorted((d for d in runs.iterdir() if d.is_dir()), key=lambda d: d.name) if runs.exists() else []
    return render([summarize_run(d) for d in dirs], str(target))
