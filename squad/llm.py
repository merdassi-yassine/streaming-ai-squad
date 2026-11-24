"""Accès au LLM, isolé ici pour pouvoir changer de fournisseur sans toucher au reste.

- Relance avec délai croissant sur les erreurs temporaires (429, 5xx).
- Repli sur le modèle suivant de la liste si un modèle reste indisponible.
- Chaque appel est archivé : entrée envoyée, réponse BRUTE, modèle réellement utilisé, jetons, erreurs.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .tracing import Tracer

REQUEST_TIMEOUT_MS = 90_000  # une requête bloquée plus de 90 s est abandonnée puis relancée
RETRYABLE = {429, 500, 502, 503, 504, "timeout"}
FATAL = {401, 403}  # problème de clé : inutile d'essayer un autre modèle


class ModelUnavailable(Exception):
    def __init__(self, errors: list[dict]):
        super().__init__(f"aucun modèle disponible : {errors}")
        self.errors = errors


def call_with_fallback(
    models: list[str],
    fn: Callable[[str], Any],
    *,
    attempts: int = 3,
    base_delay: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    on_error: Callable[[dict], None] = lambda e: None,
) -> tuple[str, Any, list[dict]]:
    """Appelle fn(model) ; renvoie (modèle utilisé, résultat, erreurs rencontrées).

    Une coupure réseau ou un délai dépassé compte comme une erreur temporaire (code "timeout").
    """
    errors: list[dict] = []
    for model in models:
        for attempt in range(1, attempts + 1):
            try:
                return model, fn(model), errors
            except Exception as e:  # noqa: BLE001 — on trie ci-dessous par code HTTP
                code = "timeout" if isinstance(e, httpx.TransportError) else getattr(e, "code", None)
                if code is None:
                    raise
                err = {"model": model, "attempt": attempt, "code": code, "message": str(e)[:300]}
                errors.append(err)
                on_error(err)
                if code in FATAL:
                    raise
                if code not in RETRYABLE:
                    break  # ex. 404 modèle fermé : on passe directement au suivant
                if attempt < attempts:
                    sleep(base_delay * 2 ** (attempt - 1))
    raise ModelUnavailable(errors)


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""


class GeminiChat:
    """Conversation avec Gemini + appels d'outils (function calling manuel, pas automatique)."""

    def __init__(self, api_key: str, models: list[str], system: str, tool_specs: list[dict],
                 tracer: Tracer, agent: str, client=None):
        from google import genai
        from google.genai import types

        self.types = types
        # Relances internes du SDK désactivées : une seule politique (la nôtre, testée) + délai maximal.
        self.client = client or genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))
        self.models, self.tracer, self.agent = models, tracer, agent
        self.system = system
        self.contents: list = []
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"])
                for t in tool_specs
            ])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def send_user(self, text: str) -> Turn:
        t = self.types
        new = t.Content(role="user", parts=[t.Part.from_text(text=text)])
        return self._generate(new)

    def send_tool_results(self, results: list[tuple[str, dict]]) -> Turn:
        t = self.types
        new = t.Content(role="user", parts=[t.Part.from_function_response(name=n, response=r) for n, r in results])
        return self._generate(new)

    def _generate(self, new_content) -> Turn:
        self.contents.append(new_content)
        start = time.time()
        model, resp, errors = call_with_fallback(
            self.models,
            lambda m: self.client.models.generate_content(model=m, contents=self.contents, config=self.config),
            on_error=lambda e: self.tracer.event(
                self.agent, f"⚠ {e['model']} essai {e['attempt']} : erreur {e['code']} → nouvel essai / modèle suivant"),
        )
        parts = []
        if resp.candidates and resp.candidates[0].content:
            self.contents.append(resp.candidates[0].content)
            parts = resp.candidates[0].content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        calls = [ToolCall(c.name, dict(c.args or {})) for c in (resp.function_calls or [])]
        usage = resp.usage_metadata
        self.tracer.record("llm_call", self.agent, {
            "models_requested": self.models,
            "model_used": model,
            "errors_before_success": errors,
            "latency_s": round(time.time() - start, 2),
            "system_prompt": self.system if len(self.contents) <= 2 else "(identique au 1er appel)",
            "input": new_content.model_dump(mode="json", exclude_none=True),
            "response_raw": resp.model_dump(mode="json", exclude_none=True, exclude={"sdk_http_response"}),
            "usage": {
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
                "thinking_tokens": getattr(usage, "thoughts_token_count", None),
            },
        })
        self.tracer.event(self.agent, f"réponse de {model} ({time.time() - start:.0f} s)")
        return Turn(text=text, tool_calls=calls, model=model)

    def transcript(self) -> list[dict]:
        return [c.model_dump(mode="json", exclude_none=True) for c in self.contents]


def make_chat(settings, system: str, tool_specs: list[dict], tracer: Tracer, agent: str):
    if settings.provider == "gemini":
        return GeminiChat(settings.api_key, settings.models, system, tool_specs, tracer, agent)
    raise NotImplementedError(settings.provider)
