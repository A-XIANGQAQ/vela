"""V09 (audit 2026-11-09): an error half-way no longer leaves the game half set up.
The backend changes files and state first and returns Steam's new launch options last; after an exception the panel writes nothing.
A  first setup: the files went in, then the live file could not be written (/tmp full): the panel showed On, Steam's options had no Vela
   token, nothing ran. Now the setup just made is taken out again: the error leaves the game Off, as it was.
B  move of a copy setup to the overlay: s3.disable went through, the new s3.enable raised: the record was gone, the panel showed Off and
   Vela's copy-mode tokens stayed in Steam's options for ever. Now the copy setup is put back as it was (or, when that fails too, the
   answer is the options without Vela's tokens).
C  Off: the NPU layer's state was dropped before s3.disable could raise "restore check failed"; the retry no longer knew the layer's
   tokens and left them in the player's options. Now the state goes only after s3.disable is through.
Run: python -I tests/test_v09_half_way.py     (VELA_TEST_ROOT=<another plugin tree>: the old code fails)"""
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("VELA_TEST_ROOT") or os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
tmp = tempfile.mkdtemp(prefix="v09_")
decky = types.ModuleType("decky")
decky.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
decky.DECKY_USER_HOME = os.path.join(tmp, "home")
decky.DECKY_PLUGIN_SETTINGS_DIR = os.path.join(tmp, "settings")
sys.modules["decky"] = decky
sys.path.insert(0, ROOT)
import gameutil  # noqa: E402
import launchopts  # noqa: E402
import main  # noqa: E402
import npu  # noqa: E402

fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


gameutil.running_appids = lambda proc="/proc": set()
home, settings = decky.DECKY_USER_HOME, decky.DECKY_PLUGIN_SETTINGS_DIR
assets = os.path.join(tmp, "assets", "npu")
os.makedirs(assets)
open(os.path.join(assets, npu.LIB), "wb").write(b"so")
shutil.copy(os.path.join(ROOT, "assets", "npu", npu.MANIFEST), assets)


def real_npu():
    return npu.Npu(assets, home, settings, log=lambda *_: None, proc_net_unix=os.path.join(tmp, "none"), systemctl=lambda u: "inactive",
                   graph_dir=os.path.join(tmp, "graphs"))


class Quiet:
    def __getattr__(self, name):
        return lambda *a, **k: None


# ---- C. Off: s3.disable raises once ("restore check failed"), then the player presses Off again
N = real_npu()
with_layer = N.enable("100", "MANGOHUD=1 %command%")
tokens = N.tokens("100")
assert all(t in launchopts.words(with_layer) for t in tokens)
calls = {"n": 0}


def s3_disable(appid, launch):
    calls["n"] += 1
    if calls["n"] == 1:
        raise RuntimeError("restore check failed")
    return {"launch_options": launch, "resolution": None}


P = main.Plugin()
P.npu, P.live, P.floor, P.cfg = N, Quiet(), Quiet(), Quiet()
P.s3 = types.SimpleNamespace(disable=s3_disable)
P._sync_gamescope = lambda *a: None
try:
    P._s3_disable("100", with_layer)
    raised = False
except RuntimeError:
    raised = True
check("C1 the first Off fails with the restore error and the NPU layer's record is still there", raised and bool((N.read("100") or {}).get("on")))
out = P._s3_disable("100", with_layer)["launch_options"]
check(f"C2 the second Off takes the layer's tokens out of the player's options ({out!r})", out == "MANGOHUD=1 %command%" and not (N.read("100") or {}).get("on"))

# ---- B. the move to the overlay: the new enable fails
N2 = real_npu()
copy_opts = N2.enable("200", 'WINEDLLOVERRIDES="dxgi=n,b" MANGOHUD=1 %command%')
log = []


class S3B:
    fail_copy = False

    def where(self, appid):
        return "d"

    def read_state(self, d):
        return {"opts": {"x": 1}, "launch_prev": "MANGOHUD=1 %command%", "launch_set": 'WINEDLLOVERRIDES="dxgi=n,b" MANGOHUD=1 %command%'}

    def disable(self, appid, launch):
        log.append("disable")
        return {"launch_options": "MANGOHUD=1 %command%", "resolution": None}      # the game's files are back, Vela's tokens out

    def setup_mode(self, appid, launch):
        return "launcher"

    def kit_options(self, appid):
        return "%command%"

    def enable(self, appid, opts, launch, resolution, mode=None):
        log.append(f"enable {mode}")
        if mode != "copy" or self.fail_copy:
            raise OSError(28, "No space left on device")
        return {"launch_options": 'WINEDLLOVERRIDES="dxgi=n,b" ' + launch}

    def log(self, *a):
        pass


P2 = main.Plugin()
P2.npu, P2.s3 = N2, S3B()
P2._kit_out = lambda appid, launch, kit, mode: launch
try:
    out = P2._overlay_migrate("200", copy_opts)
    err = None
except Exception as e:  # noqa: BLE001
    out, err = None, e
check(f"B1 the overlay setup fails: no error for the panel, the copy setup is put back ({log})", err is None and log == ["disable", "enable launcher", "enable copy"])
check(f"B2 ... with Vela's tokens and the NPU layer's in the options again, as before the move ({out!r})",
      out is not None and 'WINEDLLOVERRIDES="dxgi=n,b"' in out and all(t in launchopts.words(out) for t in N2.tokens("200")) and "MANGOHUD=1" in out)
N3 = real_npu()
copy3 = N3.enable("201", 'WINEDLLOVERRIDES="dxgi=n,b" MANGOHUD=1 %command%')
P3 = main.Plugin()
P3.npu, P3.s3 = N3, S3B()
P3.s3.fail_copy = True
P3._kit_out = lambda appid, launch, kit, mode: launch
try:
    out3 = P3._overlay_migrate("201", copy3)
    err3 = None
except Exception as e:  # noqa: BLE001
    out3, err3 = None, e
check(f"B3 the copy setup cannot be put back either: the answer is the options without any Vela token ({out3!r})", err3 is None and out3 == "MANGOHUD=1 %command%")

# ---- A. a first setup: the files go in, then the live file cannot be written
state = {"installed": False, "undone": 0}
P4 = main.Plugin()
P4.s3 = types.SimpleNamespace(scan=lambda appid, fresh=False: {}, status=lambda appid, *a, **k: {"installed": state["installed"], "opts": {}})
P4.ag = types.SimpleNamespace(status=lambda *a, **k: {"installed": False})
P4.npu = types.SimpleNamespace(status=lambda appid: {})
P4._check = lambda *a, **k: {"kind": "fsr4"}
P4._models = lambda kind, s3, vulkan=False: [("fsr4", True)]
P4._win_restart = set()


def s3_enable(appid, opts, launch, resolution, mode=None):
    state["installed"] = True
    return {"launch_options": "VELA=1 " + launch, "resolution": None}


def s3_undo(appid, launch):
    state["installed"] = False
    state["undone"] += 1
    return {"launch_options": "%command%", "resolution": None}


P4._s3_enable, P4._s3_disable = s3_enable, s3_undo
P4._launch_keep = lambda appid, launch: launch


class LiveFull:
    def set(self, appid, values):
        raise OSError(28, "No space left on device")


P4.live = LiveFull()
try:
    P4._apply_model_opts("300", "fsr4", "%command%", None)
    err4 = None
except OSError as e:
    err4 = e
check("A1 the error reaches the panel", err4 is not None)
check(f"A2 the first setup is taken out again: the game is Off, as it was (undo calls {state['undone']})", state["undone"] == 1 and state["installed"] is False)
# a game that was already set up keeps its setup when a later step fails (only a FIRST setup is undone)
state.update(installed=True, undone=0)
try:
    P4._apply_model_opts("300", "fsr4", "VELA=1 %command%", None)
except OSError:
    pass
check("A3 a game that was set up before stays set up when the live write fails", state["undone"] == 0 and state["installed"] is True)
print("test_v09_half_way " + ("OK" if not fails else f"FAIL {len(fails)}: {fails}"))
raise SystemExit(1 if fails else 0)
