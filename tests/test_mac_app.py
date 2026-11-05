"""Tests for the packaged (Mac) app's plumbing: how steps are started without a separate Python,
the start-up file's --run, and the Mac's own folder dialog. The app itself is built and tried on
a real Mac by .github/workflows/build.yml."""
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def test_steps_start_with_python_or_as_the_app_itself(monkeypatch):
    from engine import config
    cmd = config.script_command("prepare.py", Path("/photos/Sharma"), "--x")
    assert cmd == [sys.executable, str(ROOT / "scripts" / "prepare.py"), str(Path("/photos/Sharma")), "--x"]
    monkeypatch.setattr(config, "FROZEN", True)          # the packaged app: no separate Python
    assert config.script_command("cull", "/photos/Sharma", "--shortlist") == [
        sys.executable, "--run", "cull", "/photos/Sharma", "--shortlist"]


def test_the_start_up_file_runs_a_step_and_passes_its_exit_code(tmp_path):
    main = ROOT / "app" / "keepsake_main.py"
    shoot = tmp_path / "Sample Wedding" / "Raw"
    r = subprocess.run([sys.executable, str(main), "--run", "make_sample_shoot", str(shoot)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert len(list(shoot.rglob("*.JPG"))) >= 25
    r = subprocess.run([sys.executable, str(main), "--run", "no_such_step"], capture_output=True, text=True)
    assert r.returncode == 2 and "Unknown step" in r.stdout
    r = subprocess.run([sys.executable, str(main), "--run", "../engine/config"], capture_output=True, text=True)
    assert r.returncode == 2                                    # only KeepSake's own steps


def test_the_mac_window_uses_the_system_folder_dialog(monkeypatch, tmp_path):
    import keepsake_app as app
    asked = {}

    class Window:
        def create_file_dialog(self, kind, directory=""):
            asked.update(kind=kind, directory=directory)
            return ("/Volumes/Wedding Drive/Sharma",)

    fake = types.SimpleNamespace(FileDialog=types.SimpleNamespace(FOLDER="folder"))
    monkeypatch.setitem(sys.modules, "webview", fake)
    monkeypatch.setattr(app, "WINDOW", Window())
    (tmp_path / "Output").mkdir()
    picked = app.choose_folder("Choose where to save", str(tmp_path / "Output" / "Sharma Wedding"))
    assert picked == "/Volumes/Wedding Drive/Sharma"
    assert asked == {"kind": "folder", "directory": str(tmp_path / "Output")}   # opens where it would go

    fake_old = types.SimpleNamespace(FOLDER_DIALOG=20)                      # older pywebview
    monkeypatch.setitem(sys.modules, "webview", fake_old)
    app.choose_folder("x", "")
    assert asked["kind"] == 20


def test_the_packaged_app_without_a_window_asks_for_a_typed_path(monkeypatch):
    import keepsake_app as app
    monkeypatch.setattr(app, "WINDOW", None)
    monkeypatch.setattr(app, "FROZEN", True)
    assert app.choose_folder("Choose the wedding's photo folder") == ""


def test_the_package_recipe_puts_the_screens_where_the_app_looks():
    import keepsake_app as app
    from engine import config
    assert app.UI_DIR == config.ROOT / "app" / "ui" and (app.UI_DIR / "index.html").exists()
    spec = (ROOT / "packaging" / "KeepSake.spec").read_text()
    assert '"app" / "ui"), "app/ui")' in spec and '"tools/models"' in spec and '"tools/exiftool"' in spec
    assert '"engine" / "learn.py"), "engine")' in spec and '"engine" / "shots.py"), "engine")' in spec
    assert (ROOT / "packaging" / "VERSION").read_text().strip()
