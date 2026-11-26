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


def _reads(paths):
    return [ok({"content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": p})}}]})
        for i, p in enumerate(paths)]


def test_relecture_dun_meme_fichier_dedoublonnee(tmp_path):
    """Seule la dernière lecture d'un même fichier est renvoyée ; les autres fichiers restent complets."""
    http = FakeHttp(_reads(["a.py", "b.py", "a.py"]) + [ok({"content": "fin"})])
    chat = GroqChat("k", ["m"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    chat.send_user("x")
    for p in ["a.py", "b.py", "a.py"]:
        chat.send_tool_results([("read_file", {"ok": True, "content": p * 1000})])
    tools = [m for m in http.bodies[-1]["messages"] if m["role"] == "tool"]
    assert "identique fourni plus loin" in tools[0]["content"]          # 1re lecture de a.py
    assert len(tools[1]["content"]) > 3000 and len(tools[2]["content"]) > 3000  # b.py et dernière lecture de a.py
    assert all(len(m["content"]) > 3000 for m in chat.transcript() if m["role"] == "tool")  # trace intacte


def test_historique_trop_gros_reduit_par_le_debut(tmp_path):
    http = FakeHttp(_reads(["a", "b", "c", "d"]) + [ok({"content": "fin"})])
    chat = GroqChat("k", ["m"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    chat.send_user("x")
    for p in "abcd":
        chat.send_tool_results([("read_file", {"ok": True, "content": p * 9000})])
    tools = [m for m in http.bodies[-1]["messages"] if m["role"] == "tool"]
    assert "retiré (quota)" in tools[0]["content"]
    assert sum(len(m.get("content") or "") for m in http.bodies[-1]["messages"]) <= 16_000 + 100
    assert len(tools[-1]["content"]) > 9000  # le plus récent reste complet


def test_appel_outil_invalide_est_regenere(tmp_path, monkeypatch):
    """HTTP 400 tool_use_failed (outil inventé, JSON cassé) : on régénère au lieu d'abandonner le modèle."""
    monkeypatch.setattr("squad.llm.time.sleep", lambda s: None)
    bad = FakeResp(400, {"error": {"code": "tool_use_failed", "message": "attempted to call tool 'repo_browser.x'"}})
    http = FakeHttp([bad, ok({"content": "ok"})])
    chat = GroqChat("k", ["m1", "m2"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    assert chat.send_user("x").model == "m1"  # même modèle, 2e tentative


def test_rapport_resume_un_run(tmp_path):
    from squad.report import build_report
    run = tmp_path / "ai-trace" / "runs" / "T9-1"
    run.mkdir(parents=True)
    recs = [
        {"kind": "task", "task": {"id": "T9", "title": "Essai", "role": "dev", "write_allow": ["x/"]}},
        {"kind": "llm_call", "model_used": "m1", "errors_before_success": [{"code": 429}], "usage": {"input_tokens": 10}},
        {"kind": "agent_result", "iteration": 1, "files_written": ["x/a.py"]},
        {"kind": "test_run", "iteration": 1, "exit_code": 0, "tail": "=== 3 passed in 1.0s ==="},
        {"kind": "review", "iteration": 1, "review": "Recommandation : APPROUVER — ok"},
        {"kind": "decision", "iteration": 1, "decision": "approve", "feedback": "", "decided_by": "PO"},
        {"kind": "result", "result": "merged"},
    ]
    for i, r in enumerate(recs, 1):
        (run / f"{i:03d}_a_{r['kind']}.json").write_text(json.dumps(r), encoding="utf-8")
    md = build_report(tmp_path)
    assert "| `T9-1` | T9 Essai | 1 | ✓ 3 passed in 1.0s | APPROUVER | **merged** |" in md
    assert "429 ×1" in md and "**approve** (PO)" in md


def test_fichiers_deja_ecrits_resumes_dans_l_historique(tmp_path):
    big = "ligne\n" * 300
    write = lambda i: ok({"content": None, "tool_calls": [{"id": f"w{i}", "type": "function", "function": {
        "name": "write_file", "arguments": json.dumps({"path": f"f{i}.py", "content": big})}}]})
    http = FakeHttp([write(0), write(1), ok({"content": "fin"})])
    chat = GroqChat("k", ["m"], "s", SPECS, Tracer(tmp_path, "r", quiet=True), "a", http=http)
    chat.send_user("x")
    chat.send_tool_results([("write_file", {"ok": True})])
    chat.send_tool_results([("write_file", {"ok": True})])
    sent = [m for m in http.bodies[-1]["messages"] if m["role"] == "assistant"]
    first = json.loads(sent[0]["tool_calls"][0]["function"]["arguments"])
    assert "300 lignes" in first["content"] and first["path"] == "f0.py"
    last = json.loads(sent[-1]["tool_calls"][0]["function"]["arguments"])
    assert last["content"] == big  # le dernier appel reste intact
    assert big in json.dumps(chat.transcript())  # la trace garde tout
