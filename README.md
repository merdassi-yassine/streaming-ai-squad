# streaming-ai-squad

Environnement de développement multi-agents (LangGraph) qui développe le projet `streaming-platform`
sous la supervision du Project Owner. **Ce n'est pas une fonctionnalité de la plateforme.** Tout tourne en local ;
seuls les appels au LLM (API Gemini, offre gratuite) sortent du PC.

## Installation
```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # puis mettre la clé dans .env (jamais commité)
```

## Utilisation
```
python -m squad run tasks/T000-hello.yaml --target sandbox     # s'arrête à la validation humaine
python -m squad show <run_id>                                  # revoir le résumé et le diff
python -m squad decide <run_id> approve                        # fusionner dans main
python -m squad decide <run_id> changes --feedback "..."       # renvoyer à l'agent
python -m squad decide <run_id> reject                         # abandonner (la trace est gardée)
python -m squad graph                                          # graphe Mermaid pour le rapport
```

## Organisation
| Fichier | Rôle |
|---|---|
| `squad/graph.py` | Graphe LangGraph : prepare → developer → human_review → merge/discard |
| `squad/agent.py` | Boucle d'un agent (le LLM demande des outils, on les exécute) |
| `squad/workspace.py` | Outils contrôlés : list/read/write, limités au dépôt cible et aux dossiers autorisés |
| `squad/llm.py` | Appels au LLM : délai max 90 s, relances, modèles de secours, archivage des réponses brutes |
| `squad/prompts.py` + `prompts/` | Prompts versionnés avec les 5 sections Context/Persona/Task/Constraints/Examples |
| `squad/tracing.py` | Traces JSON dans `<cible>/ai-trace/runs/<run_id>/` + affichage en direct |
| `squad/gitops.py` | Branche par tâche, commits signés par l'agent, fusion seulement après approbation |
| `tasks/` | Définition des tâches (YAML) |
| `tests/` | Tests sans réseau (faux LLM) : sécurité des outils, prompts, relances, graphe complet |

## Garanties
- Aucune intégration dans `main` sans `decide ... approve`.
- Les agents n'ont ni shell, ni Git, ni accès à `.env`/`.git`, ni écriture hors des dossiers de la tâche.
- Un résultat de test n'est jamais déclaré par le LLM (pas d'exécution de code en phase 2).
- Tout est archivé : prompt rendu, entrées, réponses brutes, modèle réellement utilisé, appels d'outils, décisions.
