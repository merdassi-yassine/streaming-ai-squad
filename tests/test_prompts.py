from pathlib import Path

import pytest

from squad import prompts

ROOT = Path(__file__).resolve().parents[1]

VALID = """---
id: demo
version: 2
---
## Context
contexte {{x}}
## Persona
persona
## Task
tâche {{x}}
## Constraints
contraintes
## Examples
exemples
"""


def test_tous_les_prompts_du_depot_ont_les_5_sections():
    files = list((ROOT / "prompts").glob("*.md"))
    assert files
    for f in files:
        tpl = prompts.load(f)
        assert set(prompts.REQUIRED) <= set(tpl.sections)


def test_rendu_system_et_user():
    p = prompts.render(prompts.parse(VALID), x="42")
    assert (p.template_id, p.template_version) == ("demo", "2")
    assert p.user == "## Task\ntâche 42"
    for s in ("Persona", "Context", "Constraints", "Examples"):
        assert f"## {s}" in p.system
    assert "contexte 42" in p.system


def test_section_manquante_refusee():
    with pytest.raises(prompts.PromptError, match="Examples"):
        prompts.parse(VALID.replace("## Examples\nexemples\n", ""))


def test_variable_non_fournie_refusee():
    with pytest.raises(prompts.PromptError, match="x"):
        prompts.render(prompts.parse(VALID))
