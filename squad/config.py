"""Configuration lue depuis .env (jamais commité)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SQUAD_ROOT = Path(__file__).resolve().parents[1]
STATE_DB = SQUAD_ROOT / ".squad-state" / "checkpoints.sqlite"

# flash-lite d'abord : le 17/11/2026, les modèles « flash » de l'offre gratuite renvoyaient 503/504 en boucle
# (2 à 5 min perdues par étape) alors que flash-lite répondait. Surchargeable via GEMINI_MODELS dans .env.
DEFAULT_GEMINI_MODELS = "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.8-flash"


@dataclass(frozen=True)
class Settings:
    provider: str
    models: list[str]  # ordre de préférence ; les suivants servent de repli
    api_key: str


def get_settings() -> Settings:
    load_dotenv(SQUAD_ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider == "gemini":
        models = os.getenv("GEMINI_MODELS", DEFAULT_GEMINI_MODELS).split(",")
        key = os.getenv("GEMINI_API_KEY", "")
    else:
        raise NotImplementedError(f"Fournisseur '{provider}' pas encore pris en charge (prévu plus tard).")
    if not key or key == "colle-ta-cle-ici":
        raise RuntimeError("Clé API absente : renseigne-la dans le fichier .env")
    return Settings(provider=provider, models=[m.strip() for m in models if m.strip()], api_key=key)
