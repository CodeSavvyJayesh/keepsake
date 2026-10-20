# KeepSake desktop app

    ks app

Opens KeepSake in its own window. Choose a wedding folder (or paste its path), press
**Start culling**, and KeepSake reads every photo, chooses the best frame of every moment the way
the studio does, and copies them into `Selected` and `Not selected` (default:
`C:\KeepSake Output\<wedding>`). The photo drive is only read.

How it works: `keepsake_app.py` runs a small web server on this laptop only (127.0.0.1, with a
secret key made at every start) and opens it in a Microsoft Edge or Chrome app window. It runs the
same steps as the commands, `ks prepare` -> `ks cull` -> `ks apply`, so a stopped run carries on
where it stopped. The interface is in `ui/`: plain HTML/CSS/JS and a 3D scene written directly in
WebGL (`ui/scene.js`), so nothing is downloaded and it works offline. Fonts: Lora and Inter (SIL
Open Font License), bundled in `ui/fonts`.

Studio branding stays on the laptop, never in Git:

    data\branding\logo.png      the studio's logo, transparent background (for the dark window)
    data\branding\brand.json    {"name": "Studio name", "tagline": ""}

Without them the app shows only KeepSake's own name.
