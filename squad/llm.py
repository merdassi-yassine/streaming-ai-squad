"""Accès au LLM, isolé ici pour pouvoir changer de fournisseur sans toucher au reste.

- Relance avec délai croissant sur les erreurs temporaires (429, 5xx).
- Repli sur le modèle suivant de la liste si un modèle reste indisponible.
- Chaque appel est archivé : entrée envoyée, réponse BRUTE, modèle réellement utilisé, jetons, erreurs.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .tracing import Tracer

REQUEST_TIMEOUT_MS = 90_000  # une requête bloquée plus de 90 s est abandonnée puis relancée
RETRYABLE = {429, 500, 502, 503, 504, "timeout", "tool_use_failed"}
FATAL = {401, 403}  # problème de clé : inutile d'essayer un autre modèle
# Offre gratuite saturée (503 sur tous les modèles, vu le 24/11/2026) : au lieu d'abandonner le run,
# on fait une pause croissante (2, 4, 6, 8 min) avant une nouvelle série d'essais.
PATIENCE_ROUNDS = 5
PATIENCE_WAIT_S = 120
MAX_TOOL_CHARS = 12_000  # ~3 000 jetons : un résultat d'outil récent ne peut pas dépasser ça (Groq gratuit)


def _is_network_error(e: BaseException) -> bool:
    """Coupure réseau / délai dépassé. Le SDK Gemini embarque sa propre copie d'httpx (« httpx2 ») :
    on reconnaît donc l'erreur par le nom de ses classes, pas par isinstance (vu le 24/11/2026 :
    httpx2.ReadError [WinError 10054] non rattrapée → run interrompu)."""
    if isinstance(e, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    return any(c.__name__ in {"TransportError", "NetworkError", "TimeoutException"} for c in type(e).__mro__)


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
                code = "timeout" if _is_network_error(e) else getattr(e, "code", None)
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
        model, resp, errors = call_patiently(
            self.models,
            lambda m: self.client.models.generate_content(model=m, contents=self.contents, config=self.config),
            self.tracer, self.agent)
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


class HttpStatusError(Exception):
    """Erreur HTTP d'une API compatible OpenAI ; .code permet le tri relance / repli / fatal."""

    def __init__(self, code: int | str, body: str):
        super().__init__(f"HTTP {code} : {body[:300]}")
        self.code = code


class GroqChat:
    """Conversation avec Groq (API compatible OpenAI, offre gratuite) + appels d'outils.

    Ajouté le 24/11/2026 : l'offre gratuite de Gemini renvoyait 503 sur tous ses modèles pendant des heures.
    Même interface que GeminiChat (send_user / send_tool_results / transcript) : le reste du moteur ne change pas.
    """
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, models: list[str], system: str, tool_specs: list[dict],
                 tracer: Tracer, agent: str, http=None):
        self.http = http or httpx.Client(timeout=REQUEST_TIMEOUT_MS / 1000,
                                         headers={"Authorization": f"Bearer {api_key}"})
        self.models, self.tracer, self.agent, self.system = models, tracer, agent, system
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["parameters"]}} for t in tool_specs]
        self._pending_ids: list[str] = []

    def send_user(self, text: str) -> Turn:
        return self._generate({"role": "user", "content": text})

    def send_tool_results(self, results: list[tuple[str, dict]]) -> Turn:
        new = [{"role": "tool", "tool_call_id": cid, "name": name, "content": json.dumps(res, ensure_ascii=False)}
               for cid, (name, res) in zip(self._pending_ids, results)]
        return self._generate(*new)

    def _compacted(self) -> list[dict]:
        """Historique envoyé à l'API, réduit pour tenir sous la limite de l'offre gratuite.

        L'offre gratuite de Groq refuse une requête plus grosse que le quota de jetons par minute (HTTP 413,
        vu le 24/11/2026 : ~8 000 jetons). Les résultats d'outils déjà exploités (contenus de fichiers lus)
        sont remplacés par un court rappel ; les 2 derniers restent complets. self.messages (la trace) n'est
        pas modifié : seule la copie envoyée est réduite.
        """
        msgs = [dict(m) for m in self.messages]
        tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
        for i in tool_idx[:-2]:
            if len(msgs[i]["content"]) > 300:
                msgs[i]["content"] = json.dumps({"note": "résultat déjà exploité, retiré pour économiser le quota ; "
                                                         "relis le fichier si besoin"}, ensure_ascii=False)
        for i in tool_idx[-2:]:
            if len(msgs[i]["content"]) > MAX_TOOL_CHARS:
                msgs[i]["content"] = msgs[i]["content"][:MAX_TOOL_CHARS] + "\n…(tronqué)"
        return msgs

    def _post(self, model: str) -> dict:
        body = {"model": model, "messages": self._compacted(), "temperature": 0.2}
        if self.tools:
            body |= {"tools": self.tools, "tool_choice": "auto"}
        r = self.http.post(self.URL, json=body)
        if r.status_code == 400 and "tool_use_failed" in r.text:
            # Le modèle a généré un appel d'outil invalide (outil inventé, JSON cassé) : c'est aléatoire,
            # une nouvelle génération réussit souvent (vu le 24/11/2026 avec gpt-oss-120b).
            raise HttpStatusError("tool_use_failed", r.text)
        if r.status_code != 200:
            raise HttpStatusError(r.status_code, r.text)
        return r.json()

    def _generate(self, *new: dict) -> Turn:
        self.messages.extend(new)
        start = time.time()
        model, resp, errors = call_patiently(self.models, self._post, self.tracer, self.agent)
        msg = resp["choices"][0]["message"]
        self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls, self._pending_ids = [], []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_arguments_invalides": c["function"].get("arguments")}
            calls.append(ToolCall(c["function"]["name"], args))
            self._pending_ids.append(c["id"])
        usage = resp.get("usage") or {}
        self.tracer.record("llm_call", self.agent, {
            "provider": "groq",
            "models_requested": self.models,
            "model_used": model,
            "errors_before_success": errors,
            "latency_s": round(time.time() - start, 2),
            "system_prompt": self.system if len(self.messages) <= 3 else "(identique au 1er appel)",
            "input": list(new),
            "response_raw": resp,
            "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
        })
        self.tracer.event(self.agent, f"réponse de groq:{model} ({time.time() - start:.0f} s)")
        return Turn(text=msg.get("content") or "", tool_calls=calls, model=model)

    def transcript(self) -> list[dict]:
        return self.messages


def call_patiently(models: list[str], fn: Callable[[str], Any], tracer: Tracer, agent: str):
    """call_with_fallback + pauses croissantes si TOUS les modèles sont saturés (au lieu d'interrompre le run)."""
    errors_all: list[dict] = []
    for round_ in range(1, PATIENCE_ROUNDS + 1):
        try:
            model, resp, errors = call_with_fallback(
                models, fn, on_error=lambda e: tracer.event(
                    agent, f"⚠ {e['model']} essai {e['attempt']} : erreur {e['code']} → nouvel essai / modèle suivant"))
            return model, resp, errors_all + errors
        except ModelUnavailable as e:
            errors_all += e.errors
            if round_ == PATIENCE_ROUNDS or any(err["code"] in FATAL for err in e.errors):
                raise
            wait = PATIENCE_WAIT_S * round_
            tracer.event(agent, f"⏳ tous les modèles saturés : pause de {wait} s puis nouvelle série "
                                f"({round_ + 1}/{PATIENCE_ROUNDS})")
            time.sleep(wait)
    raise AssertionError("inaccessible")


def make_chat(settings, system: str, tool_specs: list[dict], tracer: Tracer, agent: str):
    if settings.provider == "gemini":
        return GeminiChat(settings.api_key, settings.models, system, tool_specs, tracer, agent)
    if settings.provider == "groq":
        return GroqChat(settings.api_key, settings.models, system, tool_specs, tracer, agent)
    raise NotImplementedError(settings.provider)
