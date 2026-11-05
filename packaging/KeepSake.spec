# PyInstaller recipe for the KeepSake Mac app (KeepSake.app), built by .github/workflows/build.yml:
#
#     pyinstaller --noconfirm --clean packaging/KeepSake.spec
#
# Before building, tools/models must hold the AI models (python scripts/models.py) and
# tools/exiftool the Mac ExifTool (the workflow downloads it). The studio's own data (past
# weddings, branding) is never inside the app: it lives in ~/Library/Application Support/KeepSake.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = (ROOT / "packaging" / "VERSION").read_text().strip()

for needed in ("tools/models/clip_vit_b32_vision_int8.onnx", "tools/exiftool/exiftool"):
    if not (ROOT / needed).exists():
        raise SystemExit(f"{needed} is missing: download the models and ExifTool first")

datas = [(str(ROOT / "app" / "ui"), "app/ui"),
         (str(ROOT / "tools" / "models"), "tools/models"),
         (str(ROOT / "tools" / "exiftool"), "tools/exiftool")]
binaries = []
hiddenimports = collect_submodules("scripts") + collect_submodules("engine") + ["keepsake_app"]
for package in ("rawpy", "pillow_heif", "mediapipe", "onnxruntime", "lightgbm", "webview"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "app" / "keepsake_main.py")],
    pathex=[str(ROOT), str(ROOT / "app")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="KeepSake", console=False,
          argv_emulation=False, codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, name="KeepSake")
app = BUNDLE(
    coll,
    name="KeepSake.app",
    bundle_identifier="app.keepsake.desktop",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": "KeepSake",
        "CFBundleShortVersionString": VERSION,
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSRemovableVolumesUsageDescription":
            "KeepSake reads the wedding photos on this drive. It never changes or deletes them.",
        "NSDesktopFolderUsageDescription": "KeepSake reads wedding photos and saves its folders where you choose.",
        "NSDocumentsFolderUsageDescription": "KeepSake reads wedding photos and saves its folders where you choose.",
        "NSDownloadsFolderUsageDescription": "KeepSake reads wedding photos and saves its folders where you choose.",
    },
)
