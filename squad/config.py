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
# Groq (offre gratuite, sans carte) : appel d'outils vérifié le 24/11/2026 sur ces deux modèles (< 1 s).
# qwen en dernier : il tronquait ses fichiers (25/11/2026).
DEFAULT_GROQ_MODELS = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"
# DeepSeek (payant) : modèles listés par GET /models le 26/11/2026.
DEFAULT_DEEPSEEK_MODELS = "deepseek-flash,deepseek-v4-pro"
# OpenRouter (offre gratuite, 50 requêtes/jour) : modèles « :free » avec appel d'outils, listés le 26/11/2026.
DEFAULT_OPENROUTER_MODELS = ("nvidia/nemotron-3-ultra-550b-a55b:free,google/gemma-4-31b-it:free,"
                             "nvidia/nemotron-3.5-super-vl:free")
PROVIDERS = {"gemini": ("GEMINI_MODELS", DEFAULT_GEMINI_MODELS, "GEMINI_API_KEY"),
             "groq": ("GROQ_MODELS", DEFAULT_GROQ_MODELS, "GROQ_API_KEY"),
             "deepseek": ("DEEPSEEK_MODELS", DEFAULT_DEEPSEEK_MODELS, "DEEPSEEK_API_KEY"),
             "openrouter": ("OPENROUTER_MODELS", DEFAULT_OPENROUTER_MODELS, "OPENROUTER_API_KEY")}


@dataclass(frozen=True)
class Settings:
    provider: str
    models: list[str]  # ordre de préférence ; les suivants servent de repli
    api_key: str


def get_settings(agent: str | None = None) -> Settings:
    """Fournisseur global LLM_PROVIDER, surchargeable par rôle : REVIEWER_PROVIDER=deepseek,
    FRONTEND_DEVELOPER_PROVIDER=gemini… (un modèle différent par agent = relecture plus indépendante)."""
    load_dotenv(SQUAD_ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if agent:
        provider = os.getenv(f"{agent.upper()}_PROVIDER", provider)
    if provider not in PROVIDERS:
        raise NotImplementedError(f"Fournisseur '{provider}' pas encore pris en charge.")
    models_var, default_models, key_var = PROVIDERS[provider]
    models = os.getenv(models_var, default_models).split(",")
    key = os.getenv(key_var, "")
    if not key or key == "colle-ta-cle-ici":
        raise RuntimeError(f"Clé API absente : renseigne {key_var} dans le fichier .env")
    return Settings(provider=provider, models=[m.strip() for m in models if m.strip()], api_key=key)
