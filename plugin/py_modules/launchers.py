"""0.8.23 (NONSTEAM): the game behind a non-Steam shortcut that starts a launcher instead of the game's exe.

A shortcut made by Heroic ("Add to Steam") or Lutris ("Create Steam shortcut") names the launcher's program and passes the
game in its launch options (heroic://launch/<runner>/<app>, heroic://launch?appName=<app>&runner=<runner>,
lutris:rungameid/<id>, lutris:rungame/<slug>). shortcuts.vdf has no game folder for it; the launcher's own config has:

  Heroic  <config>/heroic (native: ~/.config/heroic; flatpak: ~/.var/app/com.heroicgameslauncher.hgl/config/heroic)
            legendaryConfig/legendary/installed.json  {app: {install_path, executable, platform, title}} (Epic)
            gog_store/installed.json                  {"installed": [{appName, install_path, platform}]}  (GOG)
            nile_config/nile/installed.json           [{id, path}]                                         (Amazon)
            sideload_apps/library.json                {"games": [{app_name, install: {executable, platform}}]}
            GamesConfig/<app>.json                    {app: {winePrefix, wineVersion: {type}, targetExe}}
  Lutris  ~/.local/share/lutris (flatpak: ~/.var/app/net.lutris.Lutris/data/lutris)
            pga.db (sqlite, table games: id, slug, runner, directory, configpath)
            games/<configpath>.yml                    game: {exe, prefix, working_dir}; older Lutris: ~/.config/lutris/games

A shortcut to another launcher (Bottles, a Windows store client run through Proton) is recognised as a launcher too: its
game folder cannot be read, so the panel says so instead of setting Vela up next to the launcher's own exe.

Idea credit: the launcher list and the facts that such a game must get its files copied into the game folder (an AppImage
launcher does not start inside bwrap; a launcher that is already open starts the game outside Steam's command) are from
indiesaudi's launcher-games PRs (#3-#5, 2026-11-17), which learn the folder from the running game instead.

Pure standard library (sqlite3 included); every read is bounded and a broken or missing file gives None, never an error.
Only folders, the exe and the prefix are read: launcher configs also hold sign-in tokens, nothing else is kept or logged."""
import json
import os
import re
import sqlite3
import urllib.parse

HEROIC_FLATPAK = "com.heroicgameslauncher.hgl"
LUTRIS_FLATPAK = "net.lutris.lutris"
BOTTLES_FLATPAK = "com.usebottles.bottles"
# Windows store clients a shortcut may start through Proton (NonSteamLaunchers and the like): the game is elsewhere
WINDOWS_LAUNCHERS = {"epicgameslauncher.exe", "upc.exe", "ubisoftconnect.exe", "galaxyclient.exe", "eadesktop.exe",
                     "origin.exe", "battle.net.exe", "battle.net launcher.exe", "amazon games.exe", "itch.exe",
                     "rockstar games launcher.exe", "playnite.desktopapp.exe",
                     "playnite.fullscreenapp.exe", "legacygames.exe", "humble app.exe"}
_RX_HEROIC = re.compile(r"heroic://launch([^\s\"']*)", re.I)
_RX_LUTRIS = re.compile(r"lutris:(?://)?rungame(id)?/([^\s\"'/]+)", re.I)
MAX_JSON = 32 * 1024 * 1024  # legendary's installed.json grows with the library; anything bigger is not a config
UE_BIN = {"win64", "wingdk", "win32"}


def detect(exe, options):
    """(launcher, game id) of a shortcut that starts a launcher: ("heroic", {"runner", "app"}), ("lutris", {"id"} or
    {"slug"}), ("bottles", {}) or ("windows_launcher", {}); None for a shortcut to anything else (a game's exe, an
    emulator, a script)."""
    exe = exe or ""
    options = options or ""
    base = os.path.basename(exe.replace("\\", "/")).lower()
    low = options.lower()
    m = _RX_HEROIC.search(options)
    if m or "heroic" in base or HEROIC_FLATPAK in low:
        return "heroic", _heroic_id(m.group(1) if m else "")
    m = _RX_LUTRIS.search(options)
    if m or base == "lutris" or LUTRIS_FLATPAK in low:
        if not m:
            return "lutris", {}
        return "lutris", ({"id": m.group(2)} if m.group(1) else {"slug": urllib.parse.unquote(m.group(2))})
    if "bottles" in base or BOTTLES_FLATPAK in low or "bottles-cli" in low:
        return "bottles", {}
    if base in WINDOWS_LAUNCHERS:
        return "windows_launcher", {}
    return None


def _heroic_id(rest):
    """heroic://launch/<runner>/<app>, heroic://launch/<app> (old) or heroic://launch?appName=<app>&runner=<runner>."""
    parts = urllib.parse.urlsplit("heroic://launch" + rest)
    q = urllib.parse.parse_qs(parts.query)
    app = (q.get("appName") or q.get("appname") or [""])[0]
    runner = (q.get("runner") or [""])[0]
    path = [urllib.parse.unquote(p) for p in parts.path.split("/") if p]
    if not app and path:
        app = path[-1]
        runner = path[-2] if len(path) >= 2 else runner
    return {"runner": runner.lower(), "app": app}


def _home_path(p, home):
    p = str(p or "").strip()
    if p == "~" or p.startswith("~/"):
        p = home + p[1:]
    return os.path.normpath(p) if p else ""


def _json(path):
    try:
        if os.path.getsize(path) > MAX_JSON:
            return None
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def game_root(exe):
    """The folder to scan for a game whose exe the launcher names: the exe's folder; an Unreal game's
    <Project>/Binaries/Win64/<Project>-Win64-Shipping.exe gives the game's root (the folder with Engine/ and <Project>/,
    where the upscaler plugins are), as Steam's install folder would."""
    d = os.path.dirname(os.path.normpath(exe))
    if os.path.basename(d).lower() in UE_BIN and os.path.basename(os.path.dirname(d)).lower() == "binaries":
        project = os.path.dirname(os.path.dirname(d))
        root = os.path.dirname(project)
        return root if os.path.isdir(os.path.join(root, "Engine")) else project
    return d


def wine_prefix(prefix):
    """The Wine prefix (the folder with drive_c) of a launcher's prefix setting: Proton's compat data folder has it in pfx/."""
    if not prefix:
        return None
    if os.path.isdir(os.path.join(prefix, "pfx", "drive_c")):
        return os.path.join(prefix, "pfx")
    return prefix


def prefix_user(prefix):
    """The user folder in a Wine prefix: steamuser under Proton, else the one user folder that is not Public."""
    users = os.path.join(prefix, "drive_c", "users")
    if os.path.isdir(os.path.join(users, "steamuser")):
        return os.path.join(users, "steamuser")
    try:
        names = sorted(n for n in os.listdir(users) if n.lower() != "public" and os.path.isdir(os.path.join(users, n)))
    except OSError:
        names = []
    return os.path.join(users, names[0] if names else "steamuser")


def _game(launcher, install_dir, exe, platform, prefix, title=""):
    """The common answer: dir None when the folder is not there (not installed, a card taken out)."""
    plat = str(platform or "windows").lower()
    windows = plat in ("windows", "win", "win32", "win64") or (plat == "" and str(exe or "").lower().endswith(".exe"))
    d = install_dir if install_dir and os.path.isdir(install_dir) else None
    return {"launcher": launcher, "dir": d, "windows": windows, "exe": exe or None,
            "prefix": wine_prefix(prefix) if windows else None, "title": str(title or "")}


# ---- Heroic --------------------------------------------------------------------------------------------------------

def heroic_roots(home):
    return [os.path.join(home, ".config", "heroic"),
            os.path.join(home, ".var", "app", HEROIC_FLATPAK, "config", "heroic")]


def heroic_game(home, runner, app):
    """{"launcher", "dir", "windows", "exe", "prefix", "title"} of a Heroic game, or None when no config of this Heroic
    knows it (a removed game, a Heroic in a place of its own)."""
    if not app:
        return None
    runners = [runner] if runner in ("legendary", "gog", "nile", "sideload") else ["legendary", "gog", "nile", "sideload"]
    for root in heroic_roots(home):
        found = None
        for r in runners:
            found = _heroic_installed(home, root, r, app)
            if found:
                break
        if not found:
            continue
        install_dir, exe, platform, title = found
        gc = _json(os.path.join(root, "GamesConfig", f"{app}.json")) or {}
        gc = gc.get(app) if isinstance(gc.get(app), dict) else {}
        target = _home_path(gc.get("targetExe"), home)
        if target:
            exe = target  # the player's "Select an alternative EXE to run"
        prefix = _home_path(gc.get("winePrefix"), home) or None
        if exe and not os.path.isabs(exe) and install_dir:
            exe = os.path.join(install_dir, exe)
        if not install_dir and exe:
            install_dir = game_root(exe)
        return _game("heroic", install_dir, exe, platform, prefix, title)
    return None


def _heroic_installed(home, root, runner, app):
    """(install dir, exe or None, platform, title) from one runner's installed list, or None."""
    if runner == "legendary":
        for path in (os.path.join(root, "legendaryConfig", "legendary", "installed.json"),
                     os.path.join(home, ".config", "legendary", "installed.json")):
            data = _json(path)
            e = data.get(app) if isinstance(data, dict) else None
            if isinstance(e, dict) and e.get("install_path"):
                return (_home_path(e["install_path"], home), e.get("executable") or None, e.get("platform") or "Windows",
                        e.get("title") or "")
    elif runner == "gog":
        data = _json(os.path.join(root, "gog_store", "installed.json"))
        rows = data.get("installed") if isinstance(data, dict) else data
        for e in rows if isinstance(rows, list) else ():
            if isinstance(e, dict) and str(e.get("appName")) == app and e.get("install_path"):
                return _home_path(e["install_path"], home), None, e.get("platform") or "windows", ""
    elif runner == "nile":
        data = _json(os.path.join(root, "nile_config", "nile", "installed.json"))
        for e in data if isinstance(data, list) else ():
            if isinstance(e, dict) and str(e.get("id")) == app and e.get("path"):
                return _home_path(e["path"], home), None, "windows", ""
    elif runner == "sideload":
        data = _json(os.path.join(root, "sideload_apps", "library.json"))
        rows = data.get("games") if isinstance(data, dict) else None
        for e in rows if isinstance(rows, list) else ():
            if isinstance(e, dict) and str(e.get("app_name")) == app:
                inst = e.get("install") if isinstance(e.get("install"), dict) else {}
                exe = _home_path(inst.get("executable"), home)
                if exe:
                    return game_root(exe), exe, inst.get("platform") or "", e.get("title") or ""
    return None


# ---- Lutris --------------------------------------------------------------------------------------------------------

def lutris_roots(home):
    """(data folder with pga.db, folders with the games' .yml files)."""
    native = os.path.join(home, ".local", "share", "lutris")
    flat = os.path.join(home, ".var", "app", "net.lutris.Lutris")
    return [(native, [os.path.join(native, "games"), os.path.join(home, ".config", "lutris", "games")]),
            (os.path.join(flat, "data", "lutris"), [os.path.join(flat, "data", "lutris", "games"),
                                                    os.path.join(flat, "config", "lutris", "games")])]


def _pga_rows(db):
    """Lutris's games table as dicts (read-only, no lock taken); [] when there is none."""
    if not os.path.isfile(db):
        return []
    uri = "file:" + urllib.parse.quote(os.path.abspath(db).replace("\\", "/")) + "?mode=ro&immutable=1"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=1)
        try:
            con.row_factory = sqlite3.Row
            cols = {r[1] for r in con.execute("PRAGMA table_info(games)")}
            want = [c for c in ("id", "slug", "name", "runner", "directory", "configpath", "installed") if c in cols]
            if "id" not in want:
                return []
            return [dict(r) for r in con.execute(f"SELECT {', '.join(want)} FROM games")]
        finally:
            con.close()
    except sqlite3.Error:
        return []


def parse_yml(text):
    """The part of Lutris's game .yml Vela reads: top-level scalars and one level of scalars under each top-level key
    ({"game": {"exe": ..., "prefix": ...}, "wine": {...}, "name": ...}). Lists, deeper levels and anchors are skipped."""
    out, section = {}, None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if line.startswith("- "):
            continue
        key, sep, value = line.partition(":")
        if not sep or not key:
            continue
        key, value = key.strip().strip("'\""), _yml_scalar(value)
        if indent == 0:
            section = None
            if value == "":
                out[key] = {}
                section = key
            else:
                out[key] = value
        elif section is not None and isinstance(out.get(section), dict) and indent <= 4 and key not in out[section]:
            if value != "":
                out[section][key] = value
    return out


def _yml_scalar(v):
    v = v.strip()
    if v[:1] in ("'", '"') and v[-1:] == v[:1] and len(v) >= 2:
        q, v = v[0], v[1:-1]
        return v.replace("''", "'") if q == "'" else v.replace('\\"', '"').replace("\\\\", "\\")
    if v in ("{}", "[]", "~", "null"):
        return ""
    return v.split(" #", 1)[0].strip()


def lutris_game(home, game_id=None, slug=None):
    """{"launcher", "dir", "windows", "exe", "prefix", "title"} of a Lutris game, or None when Lutris does not know it."""
    for data_dir, yml_dirs in lutris_roots(home):
        rows = _pga_rows(os.path.join(data_dir, "pga.db"))
        row = None
        if game_id is not None:
            row = next((r for r in rows if str(r.get("id")) == str(game_id)), None)
        elif slug:
            hits = [r for r in rows if r.get("slug") == slug]
            row = next((r for r in hits if r.get("installed")), hits[0] if hits else None)
        cfg_name = (row or {}).get("configpath") or None
        yml = None
        for d in yml_dirs:
            if cfg_name and os.path.isfile(os.path.join(d, cfg_name + ".yml")):
                yml = os.path.join(d, cfg_name + ".yml")
            elif not row and slug:
                try:
                    names = sorted(n for n in os.listdir(d) if n.endswith(".yml") and
                                   (n[:-4] == slug or re.fullmatch(re.escape(slug) + r"-\d+", n[:-4])))
                except OSError:
                    names = []
                yml = os.path.join(d, names[-1]) if names else None
            if yml:
                break
        if not row and not yml:
            continue
        try:
            with open(yml, encoding="utf-8", errors="replace") as f:
                cfg = parse_yml(f.read(256 * 1024)) if yml else {}
        except (OSError, TypeError):
            cfg = {}
        game = cfg.get("game") if isinstance(cfg.get("game"), dict) else {}
        runner = str((row or {}).get("runner") or cfg.get("runner") or
                     next((k for k in ("wine", "linux") if isinstance(cfg.get(k), dict)), "")).lower()
        directory = _home_path((row or {}).get("directory"), home)
        prefix = _home_path(game.get("prefix"), home) or None
        exe = _home_path(game.get("exe"), home)
        if exe and not os.path.isabs(exe):
            bases = [b for b in (directory, prefix, _home_path(game.get("working_dir"), home)) if b]
            exe = next((os.path.join(b, exe) for b in bases if os.path.isfile(os.path.join(b, exe))),
                       os.path.join(bases[0], exe) if bases else exe)
        install_dir = game_root(exe) if exe else directory
        platform = "windows" if runner in ("wine", "proton") else "linux" if runner else ""
        if runner == "linux":
            install_dir = os.path.dirname(exe) if exe else directory
        return _game("lutris", install_dir, exe, platform, prefix if runner != "linux" else None,
                     (row or {}).get("name") or cfg.get("name") or "")
    return None


def game(home, launcher, ident):
    """The launcher's game for detect()'s answer, or None (no config for it, or a launcher Vela cannot read)."""
    if launcher == "heroic":
        return heroic_game(home, ident.get("runner", ""), ident.get("app", ""))
    if launcher == "lutris" and (ident.get("id") or ident.get("slug")):
        return lutris_game(home, ident.get("id"), ident.get("slug"))
    return None
