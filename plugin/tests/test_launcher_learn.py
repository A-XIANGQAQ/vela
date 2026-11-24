"""0.8.26 (GitHub #3): a launcher's shortcut whose game no config gives (Xodus Nexus: an AppImage; a store client in a Proton
prefix; a launcher Vela cannot read) is set up once its game has been seen running (learn.py). The panel's check says what
is missing, the files go into the game folder, a 32-bit game is refused. Steam games, shortcuts straight to an exe, Heroic /
Lutris games found by their config and emulators stay as they were.
Fake decky module and fake Steam files in a temp folder. Run: python -I tests/test_launcher_learn.py"""
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import gamecheck  # noqa: E402
import gameutil  # noqa: E402
import learn  # noqa: E402
import live  # noqa: E402
import sgsr3  # noqa: E402

tmp = tempfile.mkdtemp(prefix="vela_learn_")
NUL = bytes([0])
decky = types.ModuleType("decky")
decky.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
decky.DECKY_USER_HOME = os.path.join(tmp, "home")
decky.DECKY_PLUGIN_SETTINGS_DIR = os.path.join(tmp, "psettings")
sys.modules["decky"] = decky
sys.path.insert(0, ROOT)
import main  # noqa: E402


def s(key, value):
    return b"\x01" + key.encode() + NUL + value.encode() + NUL


def entry(index, appid, name, exe, start, options):
    return (b"\x00" + str(index).encode() + NUL + b"\x02appid" + NUL + (appid - 2 ** 32).to_bytes(4, "little", signed=True) +
            s("AppName", name) + s("Exe", f'"{exe}"') + s("StartDir", f'"{start}"') + s("LaunchOptions", options) + b"\x08")


def put(path, data=b""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


PE64 = b"MZ" + bytes(58) + (64).to_bytes(4, "little") + b"PE\0\0" + (0x8664).to_bytes(2, "little") + bytes(60) + b"d3d12.dll\0"
HIFI, EPIC, EMU, HNONE, GOG = "2641705097", "3500000003", "3877193403", "3500000001", "3123456789"
home = decky.DECKY_USER_HOME
launch = "/usr/libexec/armada/armada-game-launch %command%"
game = os.path.join(home, "Games", "Xodus Nexus", "Hi-Fi-RUSH")
exe_dir = os.path.join(game, "Hibiki", "Binaries", "WinGDK")
put(os.path.join(exe_dir, "Hi-Fi-RUSH.exe"), b"\x8f\x11 encrypted " * 100)  # not a PE file on disk
put(os.path.join(game, "Engine", "Plugins", "DLSS", "nvngx_dlss.dll"), b"x")
gog = os.path.join(home, "Games", "Witcher")
put(os.path.join(gog, "witcher3.exe"), PE64 + bytes(1024 * 1024))
celeste = os.path.join(home, "Games", "Celeste")
put(os.path.join(celeste, "Celeste.exe"), PE64)
put(os.path.join(home, "Applications", "Ryujinx.AppImage"))
steam = os.path.join(home, ".local", "share", "Steam")
put(os.path.join(steam, "userdata", "10001", "config", "shortcuts.vdf"), b"\x00shortcuts" + NUL + b"".join([
    entry(0, int(HIFI), "Hi-Fi RUSH", "/bin/sh", "/bin/", launch + ' "/h/launch-game" "4096bd52c00d45c3"'),
    entry(1, int(EPIC), "Prince of Persia", "/home/deck/Games/EpicGamesLauncher.exe", "/home/deck/Games", launch),
    entry(2, int(EMU), "Ryujinx", os.path.join(home, "Applications", "Ryujinx.AppImage"), home, launch),
    entry(3, int(HNONE), "Heroic game", "/usr/lib64/heroic/heroic", "/usr/lib64/heroic",
          launch + ' --no-gui "heroic://launch?appName=Nothing&runner=legendary"'),
    entry(4, int(GOG), "The Witcher 3", os.path.join(gog, "witcher3.exe"), gog, ""),
]) + b"\x08\x08")

live.TMP_DIR = os.path.join(tmp, "vela-live")
gameutil.running_appids = lambda proc="/proc": set()
settings = decky.DECKY_PLUGIN_SETTINGS_DIR
s3 = sgsr3.Sgsr3(os.path.join(ROOT, "assets", "sgsr3"), home, os.path.join(tmp, "s3settings"), log=lambda m: None)
s3.new_mode, s3.probe = "launcher", lambda *a: (True, "")
check = gamecheck.GameCheck(settings, log=lambda m: None)


def verdict(appid):
    sc = s3.scan(appid, fresh=True)
    return sc, check.check(sc, s3_assets=True, ag_assets=False, appid=appid)


# the shortcut's own launch options come through; what Vela adds later does not change what the shortcut starts
sc0 = sgsr3.shortcut_entry(home, HIFI)
assert sc0["options"].endswith('"4096bd52c00d45c3"') and sc0["exe"] == "/bin/sh", sc0
sig = sgsr3.shortcut_sig(sc0)
ours = f'VELA_NPU=1 {sgsr3.DLL_OVERRIDE} {sgsr3.STUDENT_ENV} OPTI_SGSR3_AMAX=0.6 '
assert sgsr3.shortcut_sig({**sc0, "options": ours + sc0["options"] + " -dx12"}) == sig
assert sgsr3.shortcut_sig({**sc0, "options": "~/.local/share/vela/bin/vela-launch " + sc0["options"]}) == sig
assert sgsr3.shortcut_sig({**sc0, "options": sc0["options"].replace("4096bd52", "7b7e70c5")}) != sig
assert sgsr3.shortcut_entry(home, "1817190") is None and sgsr3.shortcut_sig(None) is None

# which shortcuts learn at all
g = sgsr3.shortcut_game
assert sgsr3.learn_kind(g(home, HIFI)) == "linux" and sgsr3.learn_kind(g(home, HNONE)) == "launcher"
assert sgsr3.learn_kind(g(home, EMU)) == "linux" and sgsr3.learn_kind(g(home, GOG)) is None  # straight to the game's exe
assert sgsr3.learn_kind(None) is None

# nothing learned yet: the launcher's shortcut says "start once" (reason stays non_steam); an emulator reads as before
sc, v = verdict(HNONE)
assert sc["pending"] == "start_once" and (v["kind"], v["reason"], v["pending"]) == ("none", "non_steam", "start_once"), (sc, v)
assert verdict(EPIC)[1]["pending"] == "start_once"
sc, v = verdict(EMU)
assert "pending" not in sc and v["reason"] == "no_exe" and "learned" not in sc, (sc, v)
assert verdict(HIFI)[1]["kind"] == "none" and "pending" not in verdict(HIFI)[0]  # an AppImage launcher is told from an emulator only by what it runs
assert verdict(GOG)[0].get("launcher") is None and verdict(GOG)[0]["install_dir"] == gog

# Xodus Nexus: its game was seen running (the exe on disk is encrypted: the API comes from the copy that ran)
obs = {"status": "found", "certain": True, "install_dir": game, "exe": os.path.join("Hibiki", "Binaries", "WinGDK", "Hi-Fi-RUSH.exe"),
       "machine": "x64", "prefix": os.path.join(tmp, "pfx"), "run_id": HIFI}
s3.learned.record(HIFI, obs, sig=sig, markers={"dx12", "DLSS"})
sc, v = verdict(HIFI)
assert sc["install_dir"] == game and sc["exe_dir"] == exe_dir and sc["exe"] == "Hi-Fi-RUSH.exe", sc
assert sc["learned"]["confirmed"] and sc["learned"]["folder"] == "Hi-Fi-RUSH" and sc["prefix"] == obs["prefix"], sc
assert sc["launcher"] and (v["kind"], v["reason"]) == ("fsr4", None) and "dx12" in v["api"], (sc, v)
assert s3._new_mode(HIFI, "") == "copy"  # the overlay cannot hold a launcher's game
# the shortcut now starts something else: learn again
assert s3.learned.get(HIFI, "other|x") is None
put(os.path.join(home, ".local", "share", "Steam", "userdata", "10001", "config", "shortcuts.vdf"),
    open(os.path.join(steam, "userdata", "10001", "config", "shortcuts.vdf"), "rb").read().replace(b"4096bd52c00d45c3", b"ffffffffffffffff"))
assert "learned" not in s3.scan(HIFI, fresh=True) and verdict(HIFI)[1]["kind"] == "none"
put(os.path.join(steam, "userdata", "10001", "config", "shortcuts.vdf"),
    open(os.path.join(steam, "userdata", "10001", "config", "shortcuts.vdf"), "rb").read().replace(b"ffffffffffffffff", b"4096bd52c00d45c3"))

# a store client in a Proton prefix: a candidate (no id of its own) until the player picks a model; a 32-bit game is refused
esig = sgsr3.shortcut_sig(sgsr3.shortcut_entry(home, EPIC))
cobs = {"status": "found", "certain": False, "install_dir": celeste, "exe": "Celeste.exe", "machine": "x86", "prefix": None}
s3.learned.record(EPIC, cobs, sig=esig)
sc, v = verdict(EPIC)
assert sc["learned"]["confirmed"] is False and v["reason"] == "x86" and v["kind"] == "none", (sc, v)
s3.learned.record(EPIC, {**cobs, "machine": "x64"}, sig=esig)
s3.learned.confirm(EPIC)
assert s3.scan(EPIC, fresh=True)["learned"]["confirmed"] is True and verdict(EPIC)[1]["reason"] != "x86"

# a shortcut that started another shortcut's game is a launcher: its own shortcut cannot be set up
s3.learned.record(HNONE, {"status": "other", "appid": HIFI}, sig=sgsr3.shortcut_sig(sgsr3.shortcut_entry(home, HNONE)))
sc, v = verdict(HNONE)
assert v["reason"] == "non_steam" and v["pending"] == "launcher" and v["kind"] == "none", v

# a shortcut straight to the game's exe is never "learned", whatever was seen running
gsig = sgsr3.shortcut_sig(sgsr3.shortcut_entry(home, GOG))
s3.learned.record(GOG, {"status": "found", "certain": True, "install_dir": celeste, "exe": "Celeste.exe"}, sig=gsig)
sc = s3.scan(GOG, fresh=True)
assert "learned" not in sc and sc["install_dir"] == gog, sc

# the OptiScaler log of a learned game goes to Vela's own folder (a launcher may run the game from a temporary copy)
ini = s3._ini_text(s3.scan(HIFI, fresh=True), {}, exe_dir)
assert "LogFileName=Z:" in ini and (HIFI + "-OptiScaler.log") in ini
put(os.path.join(tmp, "inidir", "OptiScaler.ini"), ("[Log]" + chr(10) + "LogFileName=Z:"+chr(92)+"home"+chr(92)+"deck"+chr(92)+".config"+chr(92)+"vela"+chr(92)+"sgsr3-logs"+chr(92)+"9-OptiScaler.log" + chr(10)).encode())
assert s3._ini_log(os.path.join(tmp, "inidir")) == "/home/deck/.config/vela/sgsr3-logs/9-OptiScaler.log" and s3._ini_log(exe_dir) is None

# the plugin: a running launcher shortcut is looked at every 5 s; what is seen is kept; a game the config names is left alone
P = main.Plugin()
P.s3 = sgsr3.Sgsr3(os.path.join(tmp, "no-assets"), home, os.path.join(tmp, "p-settings"), log=lambda m: None)
P.check = check
seen = []
real_observe = learn.observe
learn.observe = lambda appid, **k: (seen.append((appid, k)) or {**obs, "handoff": True, "running_exe": None})
assert P._learn(HIFI, {HIFI}, now=100)["status"] == "found" and P.s3.learned.get(HIFI, sig)["install_dir"] == game
assert P._handoff[HIFI] is True and seen[0][1]["own_dirs"] == ["/bin", "/bin/"], seen
assert P._learn(HIFI, {HIFI}, now=102) is None and len(seen) == 1  # not again within 5 s
assert P._learn(HIFI, {HIFI}, now=106) is not None and len(seen) == 2
assert P._learn(GOG, {GOG}, now=100) is None and len(seen) == 2  # a shortcut straight to the exe: never observed
assert P._learn("12345", {"12345"}, now=100) is None  # a Steam game: never observed
assert P._learn(HIFI, set(), now=200) is None and HIFI not in P._handoff  # the game stopped
# GitHub #4: the run id of a running launcher game is kept in memory; its live file gets a copy under that id while it runs
P.live = live.Live(os.path.join(tmp, "p-settings"), log=lambda m: None)
P.live.set(HIFI, {"model": "fsr4"})
learn.observe = lambda appid, **k: {**obs, "handoff": False, "running_exe": None, "run_id": "0"}
P._learn(HIFI, {HIFI}, now=300)
P._mirror_live({HIFI})
assert P._run_ids[HIFI] == "0" and open(os.path.join(live.TMP_DIR, "0.ini")).read() == open(os.path.join(live.TMP_DIR, HIFI + ".ini")).read()
P._mirror_live(set())
assert not os.path.exists(os.path.join(live.TMP_DIR, "0.ini"))  # the game stopped: the copy goes
P._learn(HIFI, set(), now=400)
assert HIFI not in P._run_ids
learn.observe = real_observe
print("test_launcher_learn OK")
