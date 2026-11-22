"""V03 (audit 2026-11-09): the panel must not say On / Works while Vela is not in the game.
Before: (a) nothing compared Steam's real launch options with what Vela put in: options pasted over the old ones started the game without Vela
and the panel stayed green; (b) launch.json (game_dir) was written only by enable, so after the game was moved to another library vela-launch
left the kit out of every launch; (d) the status took any log newer than the setup, so a run without Vela showed "Works" from the previous
run's log.
Now: the panel sends Steam's options with the first state of each opening and the backend answers launch_fix when its launcher token / NPU
tokens are gone (the existing fix puts them back); refresh_dll makes launch.json follow the game folder; a log older than the run's start
is not shown (the panel then says "from the next start", yellow).
Run: python -I tests/test_v03_not_in_game.py     (VELA_TEST_ROOT=<another plugin tree>: the old code fails)"""
import json
import os
import shutil
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VELA_TEST_ROOT") or os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import gameutil  # noqa: E402
import launchopts  # noqa: E402
import sgsr3  # noqa: E402
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


tmp = tempfile.mkdtemp(prefix="v03_")

# ---- 1. when the run started: /proc/<pid>/stat field 22 + btime
proc = os.path.join(tmp, "proc")
write(os.path.join(proc, "stat"), "cpu 1 2 3\nbtime 1700000000\n")
write(os.path.join(proc, "100", "cmdline"), b"/usr/bin/reaper\0SteamLaunch\0AppId=4242\0--\0game.exe")
write(os.path.join(proc, "100", "stat"), "100 (reaper) S 1 " + " ".join(["0"] * 17) + " 50000 " + " ".join(["0"] * 10))
write(os.path.join(proc, "101", "cmdline"), b"/usr/bin/reaper\0SteamLaunch\0AppId=42420\0--\0other.exe")
write(os.path.join(proc, "101", "stat"), "101 (my game) S 1 " + " ".join(["0"] * 17) + " 90000 " + " ".join(["0"] * 10))
write(os.path.join(proc, "self", "cmdline"), b"x")
if hasattr(gameutil, "run_started"):
    tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    check("1a the run's start = boot time + start ticks of its SteamLaunch process", gameutil.run_started("4242", proc) == 1700000000 + 50000 / tck)
    check("1b another appid that only starts with the same digits is not this game; a name with a space parses",
          gameutil.run_started("42420", proc) == 1700000000 + 90000 / tck)
    check("1c a game that does not run: None; an unreadable /proc: None", gameutil.run_started("7", proc) is None and gameutil.run_started("4242", os.path.join(tmp, "no")) is None)
else:
    check("1 gameutil.run_started exists", False)

# ---- an overlay setup in launcher mode (as tests/test_overlay.py)
home = os.path.join(tmp, "home")
assets = os.path.join(tmp, "assets", "sgsr3")
for a in ("OptiScaler.dll", "OptiPatcher.asi"):
    write(os.path.join(assets, a), b"asset " + a.encode())
shutil.copy(os.path.join(ROOT, "assets", "sgsr3", "OptiScaler.ini"), assets)
shutil.copy(os.path.join(ROOT, "assets", "sgsr3", "VERSION.json"), assets)
settings = os.path.join(tmp, "settings")
lib = os.path.join(home, ".local", "share", "Steam")
lib2 = os.path.join(tmp, "sdcard")


def libraries(*paths):
    write(os.path.join(lib, "steamapps", "libraryfolders.vdf"),
          '"libraryfolders" { ' + " ".join('"%d" { "path" "%s" }' % (i, p.replace("\\", "/")) for i, p in enumerate(paths)) + " }")


def game(root):
    write(os.path.join(root, "steamapps", "appmanifest_4242.acf"), '"AppState" { "appid" "4242" "name" "Ov Game" "installdir" "OvGame" }')
    g = os.path.join(root, "steamapps", "common", "OvGame")
    exe = os.path.join(g, "OvGame", "Binaries", "Win64")
    write(os.path.join(g, "OvGame.exe"), b"stub" * 100)
    write(os.path.join(exe, "OvGame-Win64-Shipping.exe"), b"x" * 2000)
    write(os.path.join(g, "Engine/Plugins/Runtime/Nvidia/DLSS/Binaries/ThirdParty/Win64/nvngx_dlss.dll"), b"dlss")
    return exe


libraries(lib)
gexe = game(lib)
s3 = sgsr3.Sgsr3(assets, home, settings, log=lambda *a: None)
s3.probe = lambda exe, w, scratch: (True, "")
s3.new_mode = "launcher"
s3.enable("4242", {}, launchopts.CMD)
kitroot = os.path.join(settings, "kits", "4242")
assert s3.install_mode("4242") == "launcher" and json.load(open(os.path.join(kitroot, "launch.json")))["game_dir"] == gexe

# ---- 2. the log of the previous run is not this run's state
st = s3.read_state(s3.where("4242"))
logp = s3._log_path(s3.where("4242"), s3.kit("4242"))
write(logp, "[Init] OptiScaler v0.9\n")
t_setup = st.get("installed_at", 0)
os.utime(logp, (t_setup + 100, t_setup + 100))          # written by a run after the setup
check("2a game closed: the last run's log is shown, as before", s3.status("4242")["log"] is not None)
try:
    cur = s3.status("4242", run_since=t_setup + 500)["log"]     # this run started later: the log is the previous run's
    check("2b the game runs since after that log was written (started without Vela): no log -> the panel says 'from the next start'", cur is None)
    check("2c a log written during this run is shown", s3.status("4242", run_since=t_setup + 50)["log"] is not None)
except TypeError:
    check("2b status takes the run's start (run_since)", False)

# ---- 3. launch.json follows the game folder (moved to the SD card)
s3.cache.clear()
shutil.move(os.path.join(lib, "steamapps", "common", "OvGame"), os.path.join(lib2, "steamapps", "common", "OvGame"))
shutil.move(os.path.join(lib, "steamapps", "appmanifest_4242.acf"), os.path.join(lib2, "steamapps", "appmanifest_4242.acf"))
libraries(lib, lib2)
gexe2 = os.path.join(lib2, "steamapps", "common", "OvGame", "OvGame", "Binaries", "Win64")
assert os.path.normpath(s3.scan("4242", True)["exe_dir"]) == os.path.normpath(gexe2), s3.scan("4242")
s3.refresh_dll("4242")
lj = json.load(open(os.path.join(kitroot, "launch.json")))
check(f"3a after the move the loop's refresh rewrites launch.json's game_dir", os.path.normpath(lj["game_dir"]) == os.path.normpath(gexe2))
check("3b its other keys stay (on, layer, files, options)", lj["on"] and lj["layer"] == os.path.join(kitroot, "files") and "dxgi.dll" in lj["files"] and lj["options"])
before = open(os.path.join(kitroot, "launch.json"), "rb").read()
s3.refresh_dll("4242")
check("3c nothing moved: launch.json is not rewritten", open(os.path.join(kitroot, "launch.json"), "rb").read() == before)

# ---- 4. Steam's options lost Vela's launcher / tokens: the panel's launch_fix is asked for and puts them back
decky = types.ModuleType("decky")
decky.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
decky.DECKY_USER_HOME = home
decky.DECKY_PLUGIN_SETTINGS_DIR = settings
sys.modules["decky"] = decky
sys.path.insert(0, ROOT)
import main  # noqa: E402

gameutil.running_appids = lambda proc="/proc": set()


class Npu:
    st = None

    def read(self, appid):
        return self.st

    def ready(self):
        return False

    def rewrite_state(self, *a):
        return False

    def disable(self, appid, launch):
        return launch


P = main.Plugin()
P.s3, P.npu = s3, Npu()
good = velakit.LAUNCHER_TOKEN + " MANGOHUD=1 %command%"
if hasattr(P, "_launch_lost"):
    check("4a options not read yet: nothing claimed", P._launch_lost("4242") is False)
    P._launch_seen = {"4242": good}
    check("4b the launcher token is there: fine", P._launch_lost("4242") is False)
    P._launch_seen = {"4242": "MANGOHUD=1 %command% -novid"}        # the player pasted new options over the old ones
    check("4c launcher mode, the token is gone: launch_fix pending", P._launch_lost("4242") is True and P._launch_pending("4242", {"installed": True}) is True)
    out = P._launch_fix_seen("4242", "MANGOHUD=1 %command% -novid")["launch_options"]
    check(f"4d launch_fix puts the launcher token back in front, the player's tokens stay ({out!r})",
          out == velakit.LAUNCHER_TOKEN + " MANGOHUD=1 %command% -novid" and P._launch_lost("4242") is False)
    P._launch_seen = {"4242": 'eval $( echo "%command%" | sed "s/a/b/" )'}
    check("4e options that wrap %command% in a shell command are not 'lost' (V02: no launcher can go in front)", P._launch_lost("4242") is False)
    # ---- 6. V03c: set up, Vela not in the launch, and launch_fix cannot mend it: the state says so (the panel: a neutral existing text)
    if hasattr(P, "_not_in_launch"):
        IN = {"installed": True}
        check("6a launcher setup + options that wrap %command%: 'wrapped'", P._not_in_launch("4242", IN, set()) == "wrapped")
        P._launch_seen = {"4242": good}
        check("6b options with the launcher token: nothing to say", P._not_in_launch("4242", IN, set()) is None)
        ll = os.path.join(kitroot, "launch.log")
        write(ll, "2026-11-10 01:00:00 kit on\n2026-11-10 02:00:00 Vela left out of this launch: game folder /x is gone\n")
        real_started = gameutil.run_started
        gameutil.run_started = lambda appid, proc="/proc": os.path.getmtime(ll) - 5        # this run began just before that line
        check("6c the launcher left the kit out of the running game: 'left_out', with its reason",
              P._not_in_launch("4242", IN, {"4242"}) == "left_out" and s3.left_out("4242", 0) == "game folder /x is gone")
        check("6d the game does not run: nothing to say", P._not_in_launch("4242", IN, set()) is None)
        gameutil.run_started = lambda appid, proc="/proc": os.path.getmtime(ll) + 100      # a later run; the line is an earlier run's
        check("6e a 'left out' line of an earlier run does not count", P._not_in_launch("4242", IN, {"4242"}) is None)
        check("6f a game that is not set up: nothing to say", P._not_in_launch("4242", {"installed": False}, {"4242"}) is None)
        gameutil.run_started = real_started
        P._launch_seen = {"4242": 'eval $( echo "%command%" | sed "s/a/b/" )'}
    else:
        check("6 Plugin._not_in_launch exists", False)
    # the owner's wording (2026-11-10): one text per state, before On / Works
    import json
    en = json.load(open(os.path.join(ROOT, "src", "i18n", "en.json"), encoding="utf-8"))
    tsx = open(os.path.join(ROOT, "src", "status.ts"), encoding="utf-8").read()  # PANELWHY: the status card's file
    check("6g the panel has the owner's text for each of the two states, before On / Works",
          'if (gs.not_in_launch) return { dot: YELLOW, text: t(gs.not_in_launch === "wrapped" ? "st.launchWrapped" : "st.notConnected") };' in tsx
          and tsx.index("if (gs.not_in_launch)") < tsx.index('t("st.onIdle"')
          and en.get("st.notConnected") == "Vela did not connect to the game. Restart the game"
          and en.get("st.launchWrapped") == "This game's launch options are not compatible with Vela")
    # a copy setup with the NPU layer's tokens
    s3.install_mode = lambda appid: "copy"
    P.npu.st = {"on": True, "tokens": ["VELA_NPU=1", "VELA_NPU_LOG_FILE=/x"], "launch_set": "VELA_NPU=1 VELA_NPU_LOG_FILE=/x %command%"}
    P._launch_seen = {"4242": "VELA_NPU=1 VELA_NPU_LOG_FILE=/x %command% -w"}
    check("4f copy setup: the layer's tokens are all there (the player added his own): fine", P._launch_lost("4242") is False)
    P._launch_seen = {"4242": "%command% -w"}
    check("4g copy setup: the layer's tokens are gone: launch_fix pending", P._launch_lost("4242") is True)
else:
    check("4 Plugin._launch_lost exists", False)
import inspect  # noqa: E402
check("5 game_state takes Steam's launch options from the panel", "launch" in inspect.signature(main.Plugin.game_state).parameters)
print("test_v03_not_in_game " + ("OK" if not fails else f"FAIL {len(fails)}: {fails}"))
raise SystemExit(1 if fails else 0)
