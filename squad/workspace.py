"""Outils contrôlés donnés aux agents : lister, lire, écrire des fichiers — et rien d'autre.

Règles de sécurité (vérifiées par tests/test_workspace.py) :
- aucun accès en dehors du dépôt cible (« ../ », chemins absolus) ;
- jamais de lecture de .env ni de .git ;
- écriture seulement dans les dossiers autorisés pour la tâche, jamais dans .git ni ai-trace ;
- taille de fichier limitée.
"""
from pathlib import Path, PurePosixPath

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "ai-trace"}
FORBIDDEN_PARTS = {".git", ".env"}
MAX_BYTES = 100_000
MAX_READS_PER_FILE = 2  # au-delà, read_file refuse : évite les boucles de relecture


class Refused(Exception):
    pass


TOOL_SPECS = [
    {
        "name": "list_files",
        "description": "Liste les fichiers du dépôt (chemins relatifs).",
        "parameters": {
            "type": "object",
            "properties": {"directory": {"type": "string", "description": "Sous-dossier, '.' par défaut"}},
        },
    },
    {
        "name": "read_file",
        "description": "Lit un fichier texte du dépôt.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Crée ou remplace entièrement un fichier texte (contenu complet).",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
]


class Workspace:
    def __init__(self, root: str | Path, write_allow: list[str]):
        self.root = Path(root).resolve()
        self.write_allow = [a.strip("/") + "/" for a in write_allow]
        self.written: list[str] = []
        self.reads: dict[str, int] = {}

    def _rel(self, path: str) -> str:
        if not path or Path(path).is_absolute() or PurePosixPath(path).is_absolute():
            raise Refused(f"chemin refusé : '{path}' (relatif au dépôt uniquement)")
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root):
            raise Refused(f"chemin refusé : '{path}' sort du dépôt")
        rel = full.relative_to(self.root).as_posix()
        if set(PurePosixPath(rel).parts) & FORBIDDEN_PARTS:
            raise Refused(f"accès interdit : '{rel}'")
        return rel

    def list_files(self, directory: str = ".") -> list[str]:
        base = self.root if directory in ("", ".") else self.root / self._rel(directory)
        out = []
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(self.root)
            if p.is_file() and not (set(rel.parts) & (SKIP_DIRS | FORBIDDEN_PARTS)):
                out.append(rel.as_posix())
        return out[:500]

    def read_file(self, path: str) -> str:
        full = self.root / self._rel(path)
        if not full.is_file():
            raise Refused(f"fichier introuvable : '{path}'")
        if full.stat().st_size > MAX_BYTES:
            raise Refused(f"fichier trop gros : '{path}'")
        return full.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        rel = self._rel(path)
        if PurePosixPath(rel).parts[0] == "ai-trace":
            raise Refused("écriture interdite dans ai-trace/")
        if not any(rel.startswith(a) for a in self.write_allow):
            raise Refused(f"écriture refusée : '{rel}' hors des dossiers autorisés {self.write_allow}")
        if len(content.encode("utf-8")) > MAX_BYTES:
            raise Refused("contenu trop gros")
        full = self.root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8", newline="\n")
        if rel not in self.written:
            self.written.append(rel)
        return rel

    def execute(self, name: str, args: dict) -> dict:
        """Point d'entrée unique des appels d'outils du LLM. Ne lève jamais : renvoie ok/erreur."""
        try:
            if name == "list_files":
                return {"ok": True, "files": self.list_files(args.get("directory", "."))}
            if name == "read_file":
                return {"ok": True, "content": self.read_file(args["path"])}
            if name == "write_file":
                return {"ok": True, "written": self.write_file(args["path"], args["content"])}
            return {"ok": False, "error": f"outil inconnu : {name}"}
        except Refused as e:
            return {"ok": False, "refused": True, "error": str(e)}
        except KeyError as e:
            return {"ok": False, "error": f"argument manquant : {e}"}
