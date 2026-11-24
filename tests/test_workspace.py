import pytest

from squad.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    (tmp_path / "README.md").write_text("bonjour", encoding="utf-8")
    return Workspace(tmp_path, ["hello/"])


def test_ecriture_autorisee(ws, tmp_path):
    r = ws.execute("write_file", {"path": "hello/a.py", "content": "x = 1\n"})
    assert r["ok"] and (tmp_path / "hello/a.py").read_text() == "x = 1\n"
    assert ws.written == ["hello/a.py"]


@pytest.mark.parametrize("path", ["README2.md", "src/a.py", "hello_bis/a.py", "ai-trace/x.json"])
def test_ecriture_hors_dossiers_autorises_refusee(ws, path):
    r = ws.execute("write_file", {"path": path, "content": "x"})
    assert r["refused"] and not ws.written


@pytest.mark.parametrize("path", ["../evil.py", "hello/../../evil.py", "C:/Windows/evil.py", "/etc/passwd"])
def test_sortie_du_depot_refusee(ws, path):
    assert ws.execute("write_file", {"path": path, "content": "x"})["refused"]
    assert ws.execute("read_file", {"path": path})["ok"] is False


@pytest.mark.parametrize("path", [".env", ".git/config", "hello/../.env"])
def test_lecture_secrets_et_git_refusee(ws, path):
    assert ws.execute("read_file", {"path": path})["refused"]


def test_list_files_cache_env_et_git(ws):
    files = ws.execute("list_files", {})["files"]
    assert files == ["README.md"]


def test_contenu_trop_gros_refuse(ws):
    assert ws.execute("write_file", {"path": "hello/big.txt", "content": "x" * 200_000})["refused"]


def test_outil_inconnu_et_argument_manquant(ws):
    assert ws.execute("rm_rf", {})["ok"] is False
    assert "manquant" in ws.execute("write_file", {"path": "hello/a.py"})["error"]
