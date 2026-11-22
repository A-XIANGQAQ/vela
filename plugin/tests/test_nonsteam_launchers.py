"""0.8.23 NONSTEAM: a non-Steam shortcut run through Proton gets the same Vela flow as a Steam game.
- a Steam shortcut to a GOG copy's exe with a Proton tool in config.vdf: folder, exe and inputs found, models offered,
  Off gives back the shortcut's launch options byte for byte and the game folder as it was (copy and overlay mode);
- an Unreal *-Win64-Shipping.exe shortcut with Steam's own Start In: the game's root is scanned (plugins in Engine/);
- Heroic (Epic through legendary, GOG, sideloaded; native and flatpak) and Lutris (pga.db + games/*.yml, rungameid and
  rungame/<slug>) shortcuts: the game folder from the launcher's config, files copied (no overlay), no -dx12 for the
  launcher, the launcher's prefix for a profile's settings file, Off exact;
- a launcher without a config entry (and Bottles, a Windows store client): "Vela cannot find this game's folder";
- emulators and a launcher's Linux game stay "a Linux game".
Fake decky module, fake Steam + launcher configs in a temp folder. Run: python -I tests/test_nonsteam_launchers.py"""
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import anygame  # noqa: E402
import gamecfg  # noqa: E402
import gamecheck  # noqa: E402
import gameutil  # noqa: E402
import latency  # noqa: E402
import launchers  # noqa: E402
import live  # noqa: E402
import npu  # noqa: E402
import sgsr3  # noqa: E402
import upscale  # noqa: E402

tmp = tempfile.mkdtemp(prefix="vela_nst_")
NUL = bytes([0])


def write(path, data=b""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


def s(key, value):
    return b"\x01" + key.encode() + NUL + value.encode() + NUL


def i32(key, value):
    return b"\x02" + key.encode() + NUL + int(value).to_bytes(4, "little", signed=True)


def entry(index, appid, name, exe, start, options=""):
    signed = appid - 2 ** 32 if appid >= 2 ** 31 else appid
    return (b"\x00" + str(index).encode() + NUL + i32("appid", signed) + s("AppName", name) + s("Exe", f'"{exe}"') +
            s("StartDir", f'"{start}"') + s("icon", "") + s("LaunchOptions", options) + b"\x00tags" + NUL + b"\x08" +
            b"\x08")


def tree(d):
    """{relative path: md5} of every file under d: the game folder before and after."""
    out = {}
    for cur, _, files in os.walk(d):
        for f in files:
            p = os.path.join(cur, f)
            out[os.path.relpath(p, d)] = hashlib.md5(open(p, "rb").read()).hexdigest()
    return out


EXE = b"MZ" + b"\0" * 62 + b"d3d12.dll\0D3D12CreateDevice\0" + b"\0" * (1100 * 1024)  # a DX12 game exe of >= 1 MB

decky = types.ModuleType("decky")
decky.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
decky.DECKY_USER_HOME = os.path.join(tmp, "home")
decky.DECKY_PLUGIN_SETTINGS_DIR = os.path.join(tmp, "psettings")
sys.modules["decky"] = decky
sys.path.insert(0, ROOT)
import main  # noqa: E402

live.TMP_DIR = os.path.join(tmp, "vela-live")
gameutil.running_appids = lambda proc="/proc": set()
home = decky.DECKY_USER_HOME
settings = decky.DECKY_PLUGIN_SETTINGS_DIR
steam = os.path.join(home, ".local", "share", "Steam")
write(os.path.join(steam, "steamapps", "libraryfolders.vdf"),
      '"libraryfolders"\n{\n "0"\n {\n  "path"  "%s"\n }\n}\n' % steam.replace("\\", "\\\\"))

# ---- the games -------------------------------------------------------------------------------------------------------
GOG, UE, HER, HGOG, HSIDE, HLIN, LUT, LSLUG, LLIN, HNONE, BOT, EPIC, EMU, BAT, BATX = (
    "3123456789", "3200000001", "3300000001", "3300000002", "3300000003", "3300000004", "3400000001", "3400000002",
    "3400000003", "3500000001", "3500000002", "3500000003", "3877193403", "3600000001", "3600000002")
witcher = os.path.join(home, "Games", "GOG Games", "The Witcher 3 Wild Hunt GOTY")
w_exe_dir = os.path.join(witcher, "bin", "x64_dx12")
write(os.path.join(w_exe_dir, "witcher3.exe"), EXE)
write(os.path.join(w_exe_dir, "nvngx_dlss.dll"), b"the game's own DLSS")
write(os.path.join(w_exe_dir, "dxgi.dll"), b"a dxgi.dll the game ships")  # Vela's copy must give it back exactly
write(os.path.join(witcher, "goggame-1207664663.info"), b"{}")
ue = os.path.join(home, "Games", "Ghostly")
ue_bin = os.path.join(ue, "Ghostly", "Binaries", "Win64")
write(os.path.join(ue_bin, "Ghostly-Win64-Shipping.exe"), EXE)
write(os.path.join(ue, "Engine", "Plugins", "Runtime", "Nvidia", "DLSS", "Binaries", "ThirdParty", "Win64",
                   "nvngx_dlss.dll"), b"dlss")
write(os.path.join(ue, "Ghostly.exe"), b"MZ stub")
# Heroic: an Epic game (legendary), a GOG game (flatpak Heroic), a sideloaded game, a Linux game
h_cfg = os.path.join(home, ".config", "heroic")
h_flat = os.path.join(home, ".var", "app", "com.heroicgameslauncher.hgl", "config", "heroic")
h_epic = os.path.join(home, "Games", "Heroic", "Ghostrunner")
write(os.path.join(h_epic, "Ghostrunner", "Binaries", "Win64", "Ghostrunner-Win64-Shipping.exe"), EXE)
write(os.path.join(h_epic, "Engine", "Plugins", "Runtime", "Nvidia", "DLSS", "Binaries", "ThirdParty", "Win64",
                   "nvngx_dlss.dll"), b"dlss")
write(os.path.join(h_epic, "Ghostrunner.exe"), b"MZ stub")
h_pfx = os.path.join(home, "Games", "Heroic", "Prefixes", "default", "Ghostrunner")  # Proton: the prefix is in pfx/
os.makedirs(os.path.join(h_pfx, "pfx", "drive_c", "users", "steamuser"))
write(os.path.join(h_cfg, "legendaryConfig", "legendary", "installed.json"), json.dumps({
    "Salt": {"app_name": "Salt", "title": "Ghostrunner", "install_path": h_epic, "platform": "Windows",
             "executable": "Ghostrunner.exe", "can_run_offline": True}}))
write(os.path.join(h_cfg, "GamesConfig", "Salt.json"), json.dumps({
    "Salt": {"winePrefix": h_pfx, "wineVersion": {"type": "proton", "name": "GE-Proton9-20"}}, "version": "v0"}))
h_gog = os.path.join(home, "Games", "Heroic", "Gwent")
write(os.path.join(h_gog, "bin", "gwent.exe"), EXE)
write(os.path.join(h_gog, "bin", "libxess.dll"), b"xess")
write(os.path.join(h_flat, "gog_store", "installed.json"), json.dumps({"installed": [
    {"appName": "1971477531", "install_path": h_gog, "platform": "windows"},
    {"appName": "1400000000", "install_path": os.path.join(home, "Games", "Heroic", "Linuxy"), "platform": "linux"}]}))
write(os.path.join(home, "Games", "Heroic", "Linuxy", "start.sh"), b"#!/bin/sh")
h_side = os.path.join(home, "Games", "Side")
write(os.path.join(h_side, "side.exe"), EXE)
write(os.path.join(h_side, "amd_fidelityfx_dx12.dll"), b"fsr 3.1")
write(os.path.join(h_cfg, "sideload_apps", "library.json"), json.dumps({"games": [
    {"app_name": "aB3dE", "title": "Side", "install": {"executable": os.path.join(h_side, "side.exe"),
                                                       "platform": "Windows"}}]}))
# Lutris: pga.db + games/<configpath>.yml; a wine game in a prefix of its own, a Linux game, one by slug (flatpak, no db)
l_data = os.path.join(home, ".local", "share", "lutris")
l_pfx = os.path.join(home, "Games", "control")
l_game = os.path.join(l_pfx, "drive_c", "Program Files", "Control")
write(os.path.join(l_game, "Control_DX12.exe"), EXE)
write(os.path.join(l_game, "nvngx_dlss.dll"), b"dlss")
os.makedirs(os.path.join(l_pfx, "drive_c", "users", "deck"))
os.makedirs(os.path.join(l_pfx, "drive_c", "users", "Public"))
write(os.path.join(l_data, "games", "control-1700000000.yml"),
      "game:\n  arch: win64\n  exe: drive_c/Program Files/Control/Control_DX12.exe\n  prefix: '%s'\n"
      "game_slug: control\nname: Control\nsystem:\n  env:\n    FOO: bar\nwine:\n  version: wine-ge-8-26\n" % l_pfx)
l_lin = os.path.join(home, "Games", "tux")
write(os.path.join(l_lin, "tux.x86_64"), b"\x7fELF")
write(os.path.join(l_data, "games", "tux-1700000001.yml"), "game:\n  exe: %s\nname: Tux\nlinux: {}\n" %
      os.path.join(l_lin, "tux.x86_64"))
os.makedirs(l_data, exist_ok=True)
con = sqlite3.connect(os.path.join(l_data, "pga.db"))
con.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, name TEXT, slug TEXT, runner TEXT, directory TEXT, "
            "configpath TEXT, installed INTEGER)")
con.execute("INSERT INTO games VALUES (7, 'Control', 'control', 'wine', ?, 'control-1700000000', 1)", (l_pfx,))
con.execute("INSERT INTO games VALUES (8, 'Tux', 'tux', 'linux', ?, 'tux-1700000001', 1)", (l_lin,))
con.commit()
con.close()
l_flat = os.path.join(home, ".var", "app", "net.lutris.Lutris", "data", "lutris", "games")
l_hz = os.path.join(home, "Games", "hz")
write(os.path.join(l_hz, "hz.exe"), EXE)
write(os.path.join(l_hz, "ffx_fsr2_api_x64.dll"), b"fsr2")
write(os.path.join(l_flat, "horizon-zero-dawn-1700000002.yml"),
      'game:\n  exe: "%s"\n  prefix: %s\nwine: {}\n' % (os.path.join(l_hz, "hz.exe").replace("\\", "\\\\"),
                                                       os.path.join(home, "Games", "hzpfx")))
# an emulator, a store client through Proton, a .bat with and without Proton
apps = os.path.join(home, "Applications")
write(os.path.join(apps, "Ryujinx.AppImage"))
epic_dir = os.path.join(steam, "steamapps", "compatdata", EPIC, "pfx", "drive_c", "Program Files (x86)", "Epic Games",
                        "Launcher", "Portal", "Binaries", "Win64")
write(os.path.join(epic_dir, "EpicGamesLauncher.exe"), EXE)
bat = os.path.join(home, "Games", "Batty")
write(os.path.join(bat, "start.bat"), b"@echo off\r\nbatty.exe\r\n")
write(os.path.join(bat, "batty.exe"), EXE)
write(os.path.join(bat, "libxess.dll"), b"xess")

HEROIC = "/usr/lib64/heroic/heroic"
ARMADA = "/usr/libexec/armada/armada-game-launch %command%"
GOG_LO = '-skipintro  --windowed "C:\\a b\\save.cfg"'  # the player's own options of the GOG shortcut, spacing and quotes
HER_LO = ARMADA + ' --no-gui "heroic://launch?appName=Salt&runner=legendary"'
LUT_LO = "lutris:rungameid/7"
write(os.path.join(steam, "userdata", "10001", "config", "shortcuts.vdf"), b"\x00shortcuts" + NUL + b"".join([
    entry(0, int(GOG), "The Witcher 3", os.path.join(w_exe_dir, "witcher3.exe"), w_exe_dir, GOG_LO),
    entry(1, int(UE), "Ghostly", os.path.join(ue_bin, "Ghostly-Win64-Shipping.exe"), ue_bin),
    entry(2, int(HER), "Ghostrunner", HEROIC, os.path.dirname(HEROIC), HER_LO),
    entry(3, int(HGOG), "Gwent", "/usr/bin/flatpak", "/usr/bin",
          'run com.heroicgameslauncher.hgl --no-gui --no-sandbox "heroic://launch/gog/1971477531"'),
    entry(4, int(HSIDE), "Side", "/opt/Heroic/heroic.AppImage", "/opt/Heroic", '--no-gui "heroic://launch/sideload/aB3dE"'),
    entry(5, int(HLIN), "Linuxy", HEROIC, "/usr/lib64/heroic", '--no-gui "heroic://launch/gog/1400000000"'),
    entry(6, int(LUT), "Control", "/usr/bin/lutris", "/usr/bin", LUT_LO),
    entry(7, int(LSLUG), "HZD", "/usr/bin/flatpak", "/usr/bin", "run net.lutris.Lutris lutris:rungame/horizon-zero-dawn"),
    entry(8, int(LLIN), "Tux", "/usr/bin/lutris", "/usr/bin", "lutris:rungameid/8"),
    entry(9, int(HNONE), "Not in Heroic", HEROIC, "/usr/lib64/heroic", '--no-gui "heroic://launch/legendary/Gone"'),
    entry(10, int(BOT), "Bottled", "/usr/bin/flatpak", "/usr/bin",
          "run --command=bottles-cli com.usebottles.bottles run -p Game -b Bottle"),
    entry(11, int(EPIC), "Epic Games", os.path.join(epic_dir, "EpicGamesLauncher.exe"), epic_dir),
    entry(12, int(EMU), "Ryujinx", os.path.join(apps, "Ryujinx.AppImage"), home),
    entry(13, int(BAT), "Batty", os.path.join(bat, "start.bat"), bat),
    entry(14, int(BATX), "Batty no Proton", os.path.join(bat, "start.bat"), bat),
]) + b"\x08\x08")
# Steam's compatibility tools: Proton for the Windows shortcuts (a shortcut's key is its unsigned appid)
write(os.path.join(steam, "config", "config.vdf"), '"InstallConfigStore"\n{\n "Software"\n {\n  "Valve"\n  {\n'
      '   "Steam"\n   {\n    "CompatToolMapping"\n    {\n' + "".join(
          f'     "{a}"\n     {{\n      "name"  "{t}"\n      "config"  ""\n      "priority"  "250"\n     }}\n'
          for a, t in ((GOG, "proton_9"), (UE, "GE-Proton9-20"), (EPIC, "proton_9"), (BAT, "proton_9"),
                       ("0", "proton_experimental"))) + '    }\n   }\n  }\n }\n}\n')
os.makedirs(os.path.join(steam, "steamapps", "compatdata", GOG, "pfx", "drive_c", "users", "steamuser"))

# ---- detection and parsing ----------------------------------------------------------------------------------------
assert launchers.detect(HEROIC, HER_LO) == ("heroic", {"runner": "legendary", "app": "Salt"})
assert launchers.detect("/usr/bin/flatpak", 'run com.heroicgameslauncher.hgl "heroic://launch/gog/12"') == \
    ("heroic", {"runner": "gog", "app": "12"})
assert launchers.detect(HEROIC, '"heroic://launch/Salt"') == ("heroic", {"runner": "", "app": "Salt"})  # old URL
assert launchers.detect(HEROIC, "") == ("heroic", {"runner": "", "app": ""})  # Heroic's own window
assert launchers.detect("/usr/bin/lutris", "lutris:rungameid/7") == ("lutris", {"id": "7"})
assert launchers.detect("/usr/bin/env", "lutris lutris:rungame/my-game") == ("lutris", {"slug": "my-game"})
assert launchers.detect("/usr/bin/flatpak", "run com.usebottles.bottles") == ("bottles", {})
assert launchers.detect("C:/x/EpicGamesLauncher.exe", "")[0] == "windows_launcher"
for exe in (os.path.join(w_exe_dir, "witcher3.exe"), os.path.join(apps, "Ryujinx.AppImage"), "/usr/bin/retroarch"):
    assert launchers.detect(exe, "-L core.so %command%") is None, exe
y = launchers.parse_yml("game:\n  exe: 'C:\\it''s.exe'\n  args: -a # note\nlist:\n  - one\nname: \"N\"\nwine: {}\n")
assert y == {"game": {"exe": "C:\\it's.exe", "args": "-a"}, "list": {}, "name": "N", "wine": {}}, y  # "wine: {}" names the runner
assert launchers.parse_yml("\x00junk: : :\n  - [\n") is not None
tools = sgsr3.compat_tools(home)
assert tools[GOG] == "proton_9" and tools[UE] == "GE-Proton9-20" and HER not in tools, tools

s3 = sgsr3.Sgsr3(os.path.join(tmp, "noassets"), home, settings, log=lambda *a: None)
check = gamecheck.GameCheck(settings, log=lambda *a: None)


def scan(a):
    return s3.scan(a, fresh=True)


def verdict(a):
    return gamecheck.verdict(scan(a), check.markers(scan(a)) if scan(a)["exe_dir"] else set(), appid=a)


# a Steam shortcut to a GOG copy's exe, Proton in config.vdf
r = scan(GOG)
assert (r["install_dir"], r["exe_dir"], r["exe"]) == (w_exe_dir, w_exe_dir, "witcher3.exe"), r
assert r["inputs"] == ["DLSS"] and r["launcher"] is None and r["compat_tool"] == "proton_9", r
assert r["prefix"] == os.path.join(steam, "steamapps", "compatdata", GOG, "pfx"), r
assert verdict(GOG)["kind"] == "fsr4", verdict(GOG)
# an Unreal Shipping exe with Steam's own Start In: the root (Engine/ has the DLSS plugin), the Shipping exe is the game
r = scan(UE)
assert r["install_dir"] == ue and r["exe_dir"] == ue_bin and r["engine"] == "Unreal" and r["inputs"] == ["DLSS"], r
# Heroic
r = scan(HER)
assert (r["install_dir"], r["launcher"], r["exe"], r["inputs"]) == (h_epic, "heroic", "Ghostrunner-Win64-Shipping.exe",
                                                                    ["DLSS"]), r
assert r["prefix"] == os.path.join(h_pfx, "pfx") and r["name"] == "Ghostrunner", r
assert verdict(HER)["kind"] == "fsr4"
r = scan(HGOG)  # flatpak Heroic, GOG runner, old URL form
assert r["install_dir"] == h_gog and r["exe"] == "gwent.exe" and r["inputs"] == ["XeSS"], r
r = scan(HSIDE)
assert r["install_dir"] == h_side and r["inputs"] == ["FSR 3.1"] and verdict(HSIDE)["kind"] == "fsr4", r
r = scan(HLIN)  # Heroic's Linux game: a Linux game, as before
assert r["install_dir"] and r["exe_dir"] is None and verdict(HLIN)["reason"] == "no_exe", r
# Lutris
r = scan(LUT)
assert (r["install_dir"], r["exe"], r["launcher"], r["inputs"]) == (l_game, "Control_DX12.exe", "lutris", ["DLSS"]), r
assert r["prefix"] == l_pfx and verdict(LUT)["kind"] == "fsr4", r
r = scan(LSLUG)  # rungame/<slug>, flatpak Lutris without pga.db: the .yml by its file name
assert r["install_dir"] == l_hz and r["inputs"] == ["FSR 2"] and r["launcher"] == "lutris", r
r = scan(LLIN)
assert r["install_dir"] == l_lin and r["exe_dir"] is None and verdict(LLIN)["reason"] == "no_exe", r
# no config entry, a launcher Vela cannot read, a store client: Vela cannot see the game's folder
for a in (HNONE, BOT, EPIC):
    r = scan(a)
    assert r["install_dir"] is None and r["launcher"], (a, r)
    v = verdict(a)
    assert v["kind"] == "none" and v["reason"] == "non_steam", (a, v)
# an emulator stays a Linux program; a .bat runs through Proton only with a tool
assert verdict(EMU)["reason"] == "no_exe" and scan(EMU)["launcher"] is None
r = scan(BAT)
assert r["install_dir"] == bat and r["exe"] == "batty.exe" and r["inputs"] == ["XeSS"], r
assert scan(BATX)["exe_dir"] is None and verdict(BATX)["reason"] == "no_exe"
# a cut config.vdf or none: no tools, never an error
assert sgsr3.compat_tools(os.path.join(tmp, "nohome")) == {}

# the overlay mode: a shortcut that runs the game itself gets it, a launcher's game gets its files copied
s3.new_mode, s3.probe = "launcher", lambda *a: (True, "")
assert s3._new_mode(GOG, GOG_LO) == "launcher" and s3._new_mode(UE, "") == "launcher"
for a in (HER, HGOG, HSIDE, LUT, LSLUG):
    assert s3._new_mode(a, "") == "copy", a

# a profile's settings file in the launcher's prefix (Heroic under Proton: pfx/.../steamuser; Lutris: the user's folder)
cfg = gamecfg.GameCfg(settings, log=lambda *a: None, home=home)
cfg.prefix_of = lambda a: scan(a).get("prefix")
e = {"root": "prefix", "file": "AppData/Local/G/Engine.ini"}
assert cfg._resolve(HER, h_epic, e) == os.path.join(h_pfx, "pfx", "drive_c", "users", "steamuser", e["file"])
assert cfg._resolve(LUT, l_game, e) == os.path.join(l_pfx, "drive_c", "users", "deck", e["file"])
assert cfg._resolve(GOG, witcher, e) == os.path.join(steam, "steamapps", "compatdata", GOG, "pfx", "drive_c", "users",
                                                     "steamuser", e["file"])
assert cfg._resolve(HER, h_epic, {"file": "x.ini"}) == os.path.join(h_epic, "x.ini")

# ---- the whole flow through the plugin: models offered, On, Off exact -----------------------------------------------
assets = os.path.join(tmp, "assets")
shutil.copytree(os.path.join(ROOT, "assets", "sgsr3"), os.path.join(assets, "sgsr3"),
                ignore=shutil.ignore_patterns("*.dll", "*.asi"))
for f in ("OptiScaler.dll", "OptiPatcher.asi", "amd_fidelityfx_upscaler_vk.dll", "amd_fidelityfx_dx12.dll"):
    write(os.path.join(assets, "sgsr3", f), b"fake " + f.encode())
write(os.path.join(assets, "staticfsr", "dxgi.dll"), b"vela interceptor")
main.ASSETS = assets


class Nop:
    def __getattr__(self, name):
        return lambda *a, **k: {"on": False}


def plugin(mode="copy"):
    P = main.Plugin()
    P.up = upscale.Upscaler(settings, home, log=lambda *a: None)
    P.up.refresh = lambda: None
    P.s3 = sgsr3.Sgsr3(os.path.join(assets, "sgsr3"), home, settings, log=lambda *a: None)
    P.s3.new_mode, P.s3.probe = mode, (lambda *a: (True, ""))
    P.ag = anygame.AnyGame(os.path.join(assets, "layer"), home, settings, log=lambda *a: None)
    P.lat = latency.Latency(home, settings, log=lambda *a: None)
    P.cfg = gamecfg.GameCfg(settings, log=lambda *a: None, home=home)
    P.cfg.prefix_of = lambda a: P.s3.scan(a).get("prefix")
    P.live = live.Live(settings, log=lambda *a: None)
    P.npu = npu.Npu(os.path.join(assets, "npu"), home, settings, log=lambda *a: None,
                    proc_net_unix=os.path.join(tmp, "nounix"), systemctl=lambda: None)
    P.floor = Nop()
    P.cpu = Nop()
    P.check = gamecheck.GameCheck(settings, log=lambda *a: None)
    P._win_restart = set()
    return P


def round_trip(P, a, lo, folder, model="fsr4"):
    """Models offered, On, a second model live, Off: the shortcut's options and the game folder exactly as before."""
    before = tree(folder)
    gs = P._game_state(a, None, lo)
    assert gs["check"]["kind"] == "fsr4", (a, gs["check"])
    offered = [m for m, ok in P._models(gs["check"]["kind"], gs["sgsr3"]) if ok]
    assert model in offered and "sgsr2" in offered, (a, offered)
    on = P._apply_model(a, model, lo, "Default")["launch_options"]
    assert on != lo and P.s3.status(a)["installed"], (a, on)
    assert P._apply_model(a, "sgsr2", on, None)["launch_options"] == on  # a model change is live: options as they are
    off = P._apply_model(a, "off", on, None)["launch_options"]
    assert off == lo, (a, lo, off)
    assert tree(folder) == before, (a, set(tree(folder)) ^ set(before))
    assert not P.s3.status(a)["installed"]
    return on


# the GOG shortcut, files copied: Vela's variables in front, the player's options byte for byte behind them
P = plugin("copy")
on = round_trip(P, GOG, GOG_LO, witcher)
assert on.startswith('WINEDLLOVERRIDES="dxgi=n,b"') and on.endswith("%command% " + GOG_LO), on
assert open(os.path.join(w_exe_dir, "dxgi.dll"), "rb").read() == b"a dxgi.dll the game ships"
# ... and through the overlay (launcher mode): only the launcher token in the options, nothing in the game folder
P = plugin("launcher")
on = round_trip(P, GOG, GOG_LO, witcher)
assert "vela-launch" in on and "WINEDLLOVERRIDES" not in on and on.endswith(GOG_LO), on
# an empty shortcut option string comes back empty
P = plugin("launcher")
round_trip(P, UE, "", ue)
# Heroic's game on a device with the overlay: files copied, Vela's variables in front of the launcher, no -dx12 at the
# end of the launcher's command line (it would go to the launcher, not the game), Off exact
P = plugin("launcher")
on = round_trip(P, HER, HER_LO, h_epic)
assert on.endswith(HER_LO) and "-dx12" not in on and "vela-launch" not in on and 'WINEDLLOVERRIDES="dxgi=n,b"' in on, on
assert P.s3.install_mode(HER) is None
# Lutris: options without %command% get it in front of them, and lose it again on Off
P = plugin("launcher")
on = round_trip(P, LUT, LUT_LO, l_pfx)
assert on.endswith("%command% " + LUT_LO), on
# a launcher without a config entry: nothing to set up, the reason is the same as a shortcut whose folder is gone
P = plugin("copy")
gs = P._game_state(HNONE, None, HER_LO)
assert gs["check"]["kind"] == "none" and gs["check"]["reason"] == "non_steam", gs["check"]
try:
    P._apply_model(HNONE, "fsr4", HER_LO, "Default")
    raise AssertionError("a game without a folder was set up")
except RuntimeError as ex:
    assert "not supported" in str(ex), ex
print("test_nonsteam_launchers OK")
