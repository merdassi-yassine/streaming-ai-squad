"""GroqChat avec un faux client HTTP (aucun appel réseau) : format OpenAI, appels d'outils, trace, erreurs."""
import json

import pytest

from squad.llm import GroqChat, ModelUnavailable
from squad.tracing import Tracer


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


class FakeHttp:
    def __init__(self, script):
        self.script, self.bodies = list(script), []

    def post(self, url, json):
        self.bodies.append(json)
        return self.script.pop(0)


def ok(message):
    return FakeResp(200, {"choices": [{"message": {"role": "assistant", **message}}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


SPECS = [{"name": "read_file", "description": "lit", "parameters": {"type": "object", "properties": {}}}]


def test_appel_outil_puis_resultat_renvoye_avec_le_bon_id(tmp_path):
    http = FakeHttp([
        ok({"content": None, "tool_calls": [{"id": "c1", "type": "function",
                                             "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]}),
        ok({"content": "Résumé final"}),
    ])
    chat = GroqChat("k", ["m1"], "système", SPECS, Tracer(tmp_path, "r", quiet=True), "backend_developer", http=http)
    t1 = chat.send_user("tâche")
    assert t1.tool_calls[0].name == "read_file" and t1.tool_calls[0].args == {"path": "a.py"}
    t2 = chat.send_tool_results([("read_file", {"ok": True, "content": "x"})])
    assert t2.text == "Résumé final" and not t2.tool_calls
    sent = http.bodies[1]["messages"]
    assert sent[0]["role"] == "system" and sent[-1]["role"] == "tool" and sent[-1]["tool_call_id"] == "c1"
    assert http.bodies[0]["tools"][0]["function"]["name"] == "read_file"
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "ai-trace/runs/r").glob("*llm_call.json")]
    assert len(traces) == 2 and traces[0]["provider"] == "groq" and traces[0]["model_used"] == "m1"


def test_repli_sur_le_2e_modele_apres_429(tmp_path, monkeypatch):
    monkeypatch.setattr("squad.llm.time.sleep", lambda s: None)
    http = FakeHttp([FakeResp(429, {"error": "rate"})] * 3 + [ok({"content": "ok"})])
    chat = GroqChat("k", ["m1", "m2"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    assert chat.send_user("x").model == "m2"


def test_cle_invalide_arrete_immediatement(tmp_path, monkeypatch):
    monkeypatch.setattr("squad.llm.time.sleep", lambda s: None)
    http = FakeHttp([FakeResp(401, {"error": "invalid key"})])
    chat = GroqChat("k", ["m1", "m2"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    with pytest.raises(Exception, match="401"):
        chat.send_user("x")
    assert len(http.bodies) == 1  # pas d'essai inutile sur un autre modèle


def test_historique_compacte_sans_toucher_la_trace(tmp_path):
    """Évite le HTTP 413 de l'offre gratuite : les vieux résultats d'outils ne sont pas renvoyés en entier."""
    big = {"ok": True, "content": "x" * 5000}
    http = FakeHttp([ok({"content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        for i in range(4)] + [ok({"content": "fin"})])
    chat = GroqChat("k", ["m"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    chat.send_user("x")
    for _ in range(4):
        chat.send_tool_results([("read_file", big)])
    sent_tools = [m for m in http.bodies[-1]["messages"] if m["role"] == "tool"]
    assert len(sent_tools) == 4
    assert all("déjà exploité" in m["content"] for m in sent_tools[:2])
    assert all(len(m["content"]) > 5000 for m in sent_tools[2:])
    assert all(len(m["content"]) > 5000 for m in chat.transcript() if m["role"] == "tool")  # trace intacte
