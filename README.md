# KeepSake

AI photo culling for wedding studios. KeepSake reads a whole shoot straight from the drive,
sorts it into **Selected** and **Removed**, explains every removal, and lets the editor
review and move any photo before anything is applied. Originals are never changed.

Status: **Phase 1 - ingest engine.** (Phase 0, setup and test data, is complete.)

## First-time setup (Windows)

Open PowerShell and run:

```powershell
cd C:\KeepSake
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

It installs Python 3.11, Node.js, Git, VS Code and ExifTool if missing, creates the Python
environment, makes the photo folders, creates the Git repository and checks everything.
Safe to run again at any time (`ks setup` does the same).

## Everyday commands

Run these inside `C:\KeepSake` (in PowerShell type `.\ks`, in Command Prompt just `ks`):

| Command | What it does |
| --- | --- |
| `ks check` | Checks every tool and this laptop; saves `docs\phase0_check.txt` |
| `ks list [folder]` | Counts photos by folder, file type and camera; finds RAW+JPEG pairs and broken files |
| `ks previews [folder]` | Pulls the preview out of each RAW file, times it, saves samples |
| `ks edge` | Creates the photo folders and the edge-case test files |
| `ks match "folder"` | Matches a wedding's selected photos to the originals; saves labels to `data\labels` |
| `ks ingest "folder"` | Reads a whole shoot once: catalog, analysis copies, thumbnails (resumes if stopped) |
| `ks ingest "folder" --bench` | Speed test on this drive; `ks ingest` then uses the best number of workers |
| `ks test` | Runs the automatic tests |

Without a folder, `list` and `previews` use `data\test_photos`.

## Ingest (Phase 1)

```
ks ingest "E:\Sharma Wedding\Raw"
```

Reads every photo once, in parallel, and writes to `data\cache\<shoot>\` on this laptop:

- `catalog.db` - SQLite: one row per photo with camera body, lens, capture time, exposure and status
- `analysis\` - an upright copy of every photo, long side 1620 px, that all later phases analyse
- `thumbs\` - 400 px thumbnails for the photo grid
- `report.txt` - photo counts, speed, camera bodies with clock warnings, files that could not be read

Most Nikon and Canon JPEGs carry a ready-made 1620 x 1080 preview at the end of the file, so
ingest reads that instead of the full photo: about a tenth of the data from the drive. Other
files are decoded at reduced size. RAW + JPEG saved together count as one photo.
Options: `--workers N`, `--limit N` (try a few), `--redo` (make every copy again).
Ctrl+C stops safely; the same command continues where it stopped.

## Folder layout

```
C:\KeepSake\
  engine\        Python AI engine (grows in Phases 1-6)
  scripts\       command-line tools behind ks.cmd
  tests\         automatic tests
  app\           Electron desktop app (Phase 5)
  docs\          notes, laptop report, speed results
  data\          PHOTOS - never in Git
    test_photos\   good, blur, eyes_closed, exposure, bursts, mixed_formats, edge_cases
    weddings\      one folder per client wedding
    scratch\       previews and reports the scripts write
    cache\         one folder per ingested shoot: catalog, analysis copies, thumbnails
    labels\        the studio's selected / not selected labels per wedding
  tools\         ExifTool (downloaded by setup) - not in Git
  .venv\         Python environment - not in Git
```

## Rules

1. **Photos never go in Git.** `data\` is ignored; setup checks that no photo is tracked.
2. **Originals are read-only.** KeepSake only reads the shoot drive and writes nothing there;
   files move only on Apply (Phase 4).
3. **Commit at the end of every session:** `git add -A` then `git commit -m "what changed"`.
4. **Measure, don't guess.** Speed numbers go in `docs\`, taken on real photos.

## Connect to GitHub (once)

1. On github.com create a **private** repository named `keepsake`, with no README.
2. In `C:\KeepSake` run:

```powershell
git remote add origin https://github.com/<your-username>/keepsake.git
git push -u origin main
```
