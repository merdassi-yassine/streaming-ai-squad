"""AI Development Squad — environnement de développement multi-agents (hors produit)."""
import os

# Tout reste local : on désactive explicitement l'envoi de traces vers LangSmith (service en ligne).
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
