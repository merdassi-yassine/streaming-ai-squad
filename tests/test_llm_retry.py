import pytest

from squad.llm import ModelUnavailable, call_with_fallback


class ApiErr(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def scripted(outcomes):
    """outcomes : {modèle: [code d'erreur ou 'ok', ...]} consommés dans l'ordre."""
    calls = []

    def fn(model):
        calls.append(model)
        o = outcomes[model].pop(0)
        if o == "ok":
            return f"réponse de {model}"
        raise ApiErr(o)

    return fn, calls


def test_relance_puis_succes_sur_le_meme_modele():
    fn, calls = scripted({"a": [503, "ok"]})
    sleeps = []
    model, res, errors = call_with_fallback(["a"], fn, sleep=sleeps.append)
    assert (model, res) == ("a", "réponse de a")
    assert len(errors) == 1 and sleeps == [5.0]


def test_repli_sur_le_modele_suivant_apres_3_echecs():
    fn, calls = scripted({"a": [503, 503, 503], "b": ["ok"]})
    sleeps = []
    model, _, errors = call_with_fallback(["a", "b"], fn, sleep=sleeps.append)
    assert model == "b" and calls == ["a", "a", "a", "b"]
    assert sleeps == [5.0, 10.0]  # délai croissant, pas d'attente inutile après le dernier essai


def test_404_passe_directement_au_modele_suivant():
    fn, calls = scripted({"a": [404], "b": ["ok"]})
    model, _, _ = call_with_fallback(["a", "b"], fn, sleep=lambda s: None)
    assert model == "b" and calls == ["a", "b"]


def test_erreur_de_cle_arrete_tout():
    fn, calls = scripted({"a": [401], "b": ["ok"]})
    with pytest.raises(ApiErr):
        call_with_fallback(["a", "b"], fn, sleep=lambda s: None)
    assert calls == ["a"]


def test_tous_les_modeles_indisponibles():
    fn, _ = scripted({"a": [503, 503, 503], "b": [429, 429, 429]})
    with pytest.raises(ModelUnavailable) as e:
        call_with_fallback(["a", "b"], fn, sleep=lambda s: None)
    assert len(e.value.errors) == 6


def test_delai_depasse_traite_comme_erreur_temporaire():
    import httpx

    state = {"n": 0}

    def fn(model):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ReadTimeout("trop long")
        return "ok"

    seen = []
    model, res, errors = call_with_fallback(["a"], fn, sleep=lambda s: None, on_error=seen.append)
    assert res == "ok" and errors[0]["code"] == "timeout" and seen == errors


def test_erreur_inattendue_non_masquee():
    def fn(model):
        raise KeyError("bug")

    with pytest.raises(KeyError):
        call_with_fallback(["a", "b"], fn, sleep=lambda s: None)
