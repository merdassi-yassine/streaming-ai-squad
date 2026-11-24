"""Modèles de prompts versionnés, avec les 5 éléments exigés : Context, Persona, Task, Constraints, Examples.

Format d'un fichier prompts/<agent>.vN.md :
    ---
    id: backend_developer
    version: 1
    ---
    ## Context
    ...
    ## Persona / ## Task / ## Constraints / ## Examples
Variables : {{nom}} (une variable inconnue est une erreur).
"""
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ["Context", "Persona", "Task", "Constraints", "Examples"]
VAR = re.compile(r"\{\{(\w+)\}\}")


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    sections: dict[str, str]
    path: str


@dataclass(frozen=True)
class RenderedPrompt:
    template_id: str
    template_version: str
    system: str  # Persona + Context + Constraints + Examples
    user: str  # Task


def parse(text: str, path: str = "<texte>") -> PromptTemplate:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text.replace("\r\n", "\n"), re.S)
    if not m:
        raise PromptError(f"{path} : en-tête --- id/version --- manquant")
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    parts = re.split(r"^## (\w+)\s*$", m.group(2), flags=re.M)
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    missing = [s for s in REQUIRED if not sections.get(s)]
    if missing:
        raise PromptError(f"{path} : sections manquantes ou vides : {missing}")
    if "id" not in meta or "version" not in meta:
        raise PromptError(f"{path} : id et version obligatoires")
    return PromptTemplate(id=meta["id"], version=meta["version"], sections=sections, path=path)


def load(path: str | Path) -> PromptTemplate:
    return parse(Path(path).read_text(encoding="utf-8"), str(path))


def _fill(text: str, variables: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        if m.group(1) not in variables:
            raise PromptError(f"variable non fournie : {{{{{m.group(1)}}}}}")
        return str(variables[m.group(1)])

    return VAR.sub(sub, text)


def render(tpl: PromptTemplate, **variables: str) -> RenderedPrompt:
    s = {k: _fill(v, variables) for k, v in tpl.sections.items()}
    system = "\n\n".join(f"## {k}\n{s[k]}" for k in ("Persona", "Context", "Constraints", "Examples"))
    return RenderedPrompt(tpl.id, tpl.version, system, f"## Task\n{s['Task']}")
