"""RSL (2026-11-22, Red Dead Redemption 2 1174180 on the console): a game Steam starts through Rockstar's launcher stub
(PlayRDR2.exe -> Rockstar Games Launcher -> RDR2.exe) showed "Unable to launch game, please verify your game files" inside
vela-launch's overlay and started fine without it. Such a game is set up in the game folder (copy mode: backup + exact
restore), an overlay setup of it made by an older Vela moves there (main._copy_migrate, game closed), and vela-launch
leaves an overlay kit of it out (the game starts as without Vela) unless VELA_STUB_OVERLAY=1 / kit "stub_overlay" asks
for the overlay (a console test of the overlay itself).
Run: python -I tests/test_rsl.py     (VELA_TEST_ROOT=<another plugin tree>: the old code fails)"""
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VELA_TEST_ROOT") or os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
tmp = tempfile.mkdtemp(prefix="vela-rsl-")
decky = types.ModuleType("decky")
decky.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
decky.DECKY_USER_HOME = os.path.join(tmp, "home")
decky.DECKY_PLUGIN_SETTINGS_DIR = os.path.join(tmp, "settings")
sys.modules["decky"] = decky
sys.path.insert(0, ROOT)
import gameutil  # noqa: E402
import launchopts  # noqa: E402
import main  # noqa: E402
import sgsr3  # noqa: E402
import vela_launch  # noqa: E402
import velakit  # noqa: E402

fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


def tree(root):
    out = {}
    for top, _, files in os.walk(root):
        for n in files:
            p = os.path.join(top, n)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = open(p, "rb").read()
    return out


class Ran:
    def __init__(self, code=0, err=b""):
        self.returncode, self.stderr = code, err


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **k: None


gameutil.running_appids = lambda proc="/proc": set()

# ---- the stub list and its detection
check("stub names are lower case", all(n == n.lower() and n.endswith(".exe") for n in vela_launch.LAUNCHER_STUBS))
check("PlayRDR2 / PlayGTAV / PlayGTAIV / PlayRDR listed",
      {"playrdr2.exe", "playgtav.exe", "playgtaiv.exe", "playrdr.exe"} <= set(vela_launch.LAUNCHER_STUBS))
d = os.path.join(tmp, "detect")
write(os.path.join(d, "PlayRDR2.exe"), b"stub")
write(os.path.join(d, "RDR2.exe"), b"game")
check("stub found, any case", vela_launch.launcher_stub(d) == "PlayRDR2.exe")
check("no folder / None: no stub", vela_launch.launcher_stub(None, os.path.join(tmp, "none")) is None)
write(os.path.join(tmp, "plain", "Game.exe"), b"g")
write(os.path.join(tmp, "plain", "PlayRDR2.exe.bak", "x"), b"")  # a folder, not the stub
check("a folder with the name is no stub", vela_launch.launcher_stub(os.path.join(tmp, "plain")) is None)

# ---- vela-launch: an overlay kit of a stub game is left out (the game runs as without Vela), unless asked for
kits = os.path.join(tmp, "kits")
game = os.path.join(tmp, "lib", "steamapps", "common", "Red Dead Redemption 2")
write(os.path.join(game, "PlayRDR2.exe"), b"stub" * 10)
write(os.path.join(game, "RDR2.exe"), b"x" * 5000)
layer = os.path.join(kits, "1174180", "files")
write(os.path.join(layer, "dxgi.dll"), b"kit dxgi")
kit = {"on": True, "game_dir": game, "layer": layer, "work_dir": os.path.join(os.path.dirname(game), ".vela-1174180"),
       "files": ["dxgi.dll"], "options": 'WINEDLLOVERRIDES="dxgi=n,b" %command%'}
write(os.path.join(kits, "1174180", "launch.json"), json.dumps(kit))
env = {"SteamAppId": "1174180"}
try:
    vela_launch.plan([os.path.join(game, "PlayRDR2.exe")], env, kits, run=lambda a, **k: Ran())
    check("stub game: overlay kit left out", False)
except RuntimeError as e:
    check("stub game: overlay kit left out, the log says why", "PlayRDR2.exe" in str(e) and "without Vela" in str(e))
planned = vela_launch.plan(["/g"], {**env, vela_launch.STUB_OVERLAY: "1"}, kits, run=lambda a, **k: Ran())
check("VELA_STUB_OVERLAY=1: the overlay as before", bool(planned) and planned[0][0] == "bwrap")
write(os.path.join(kits, "1174180", "launch.json"), json.dumps({**kit, "stub_overlay": True}))
planned = vela_launch.plan(["/g"], env, kits, run=lambda a, **k: Ran())
check("kit stub_overlay: the overlay as before", bool(planned) and planned[0][0] == "bwrap")

# ---- sgsr3: a stub game gets the copy setup (backup, exact restore), the probe is not even run
assets = os.path.join(tmp, "assets", "sgsr3")
for a in ("OptiScaler.dll", "OptiPatcher.asi"):
    write(os.path.join(assets, a), b"asset " + a.encode())
shutil.copy(os.path.join(ROOT, "assets", "sgsr3", "OptiScaler.ini"), assets)
shutil.copy(os.path.join(ROOT, "assets", "sgsr3", "VERSION.json"), assets)
home, settings = decky.DECKY_USER_HOME, decky.DECKY_PLUGIN_SETTINGS_DIR
lib = os.path.join(home, ".local", "share", "Steam")
write(os.path.join(lib, "steamapps", "libraryfolders.vdf"), '"libraryfolders" { "0" { "path" "%s" } }' % lib.replace("\\", "/"))
write(os.path.join(lib, "steamapps", "appmanifest_1174180.acf"),
      '"AppState" { "appid" "1174180" "name" "Red Dead Redemption 2" "installdir" "Red Dead Redemption 2" }')
write(os.path.join(lib, "steamapps", "appmanifest_4242.acf"), '"AppState" { "appid" "4242" "name" "Ov" "installdir" "Ov" }')
groot = os.path.join(lib, "steamapps", "common", "Red Dead Redemption 2")
write(os.path.join(groot, "PlayRDR2.exe"), b"stub" * 100)
write(os.path.join(groot, "RDR2.exe"), b"x" * (2 << 20))
write(os.path.join(groot, "nvngx_dlss.dll"), b"dlss")
write(os.path.join(lib, "steamapps", "common", "Ov", "Ov.exe"), b"y" * (2 << 20))
before = tree(groot)
logs = []
s3 = sgsr3.Sgsr3(assets, home, settings, log=logs.append)
probes = []
s3.probe = lambda exe, w, scratch: probes.append(exe) or (True, "")
s3.new_mode = "launcher"
check("stub detected from the scan", s3.launcher_stub("1174180") == "PlayRDR2.exe")
check("stub game: copy mode, no probe", s3.setup_mode("1174180", "%command%") == "copy" and probes == [])
s3.setup_mode("1174180", "%command%")
check("stub line logged once", sum("PlayRDR2.exe starts the game" in m for m in logs) == 1)
check("a game without a stub keeps the overlay", s3.setup_mode("4242", "%command%") == "launcher")
r = s3.enable("1174180", {}, launchopts.CMD)
check("copy setup: files in the game folder with state",
      s3.install_mode("1174180") == "copy" and os.path.isfile(os.path.join(groot, ".upscale-sgsr3", "state.json")) and
      open(os.path.join(groot, "dxgi.dll"), "rb").read() == b"asset OptiScaler.dll")
check("copy setup: Vela's tokens in Steam's options", 'WINEDLLOVERRIDES="dxgi=n,b"' in r["launch_options"])
check("copy setup: no move to the overlay pending", not s3.migrate_pending("1174180") and not s3.stub_move_pending("1174180"))
off = s3.disable("1174180", r["launch_options"])
check("Off: the game folder exactly as before", tree(groot) == before and off["launch_options"] == launchopts.CMD)

# ---- an overlay setup by an older Vela: main moves it to the game folder (game closed), Steam's options exact
player = "FOO=1 /usr/libexec/armada/armada-game-launch %command%"
s3.enable("1174180", {}, launchopts.CMD, mode="launcher")  # what 0.8.23 did
steam = s3.steam_options("1174180", player, True)
check("older setup: overlay, nothing in the game folder", s3.install_mode("1174180") == "launcher" and tree(groot) == before)
check("older setup: the move is pending", s3.stub_move_pending("1174180") and not s3.migrate_pending("1174180"))
P = main.Plugin()
P.s3, P.live, P.floor, P.cfg = s3, Quiet(), Quiet(), Quiet()
P.npu = types.SimpleNamespace(read=lambda a: None, ready=lambda: False, status=lambda a: {})
P.nofg = None
P._sync_gamescope = lambda *a: None
check("launch_fix is due", P._launch_pending("1174180", {"installed": True}))
out = P._copy_migrate("1174180", steam)
check("moved: copy setup", s3.install_mode("1174180") == "copy" and
      open(os.path.join(groot, "dxgi.dll"), "rb").read() == b"asset OptiScaler.dll")
check("moved: the launcher token out, the player's tokens stay, Vela's in front",
      velakit.LAUNCHER_TOKEN not in launchopts.words(out) and "FOO=1" in out and 'WINEDLLOVERRIDES="dxgi=n,b"' in out and
      "/usr/libexec/armada/armada-game-launch %command%" in out)
lj = json.load(open(os.path.join(settings, "kits", "1174180", "launch.json"), encoding="utf-8"))
check("moved: the kit is off for vela-launch", not lj.get("on"))
check("moved: nothing pending any more", not s3.stub_move_pending("1174180") and not s3.migrate_pending("1174180"))
off = s3.disable("1174180", out)
check("Off after the move: options and folder exactly as before Vela", off["launch_options"] == player and tree(groot) == before)

print("test_rsl OK" if not fails else f"test_rsl FAILED: {fails}")
sys.exit(1 if fails else 0)
