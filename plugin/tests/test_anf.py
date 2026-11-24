"""ANF (branch anf-model): the panel model "anf". Offered only with all of: the plugin's DLL has it (VERSION.json "anf_live"), the chip is the
SM8550, the NPU is ready and the system package carries an ANF family (package 18: fsr4npud_anf<canvas>s32 graph files / sockets).
Not in a Vulkan or a static (RE4 interceptor) game. live npu=2 / npuexact=1 like FSR 4. DX11 runs it on the sgsr3_12 bridge.
The status line reads "Vela ANF: ready" of the fork's log. A fresh temp folder per run. Run: python -I tests/test_anf.py"""
import os
import re
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import device  # noqa: E402
import npu  # noqa: E402
import presets  # noqa: E402
import sgsr3  # noqa: E402

tmp = tempfile.mkdtemp(prefix="vela_anf_")

# ---- presets
assert "anf" in presets.MODELS and presets.ANF_MODELS == ("anf",) and "anf" not in presets.FSR4_MODELS
assert presets.NPU_MODELS == ("fsr4", "fsr4fast", "anf")
assert dict(presets.models_for("fsr4", anf=True))["anf"] and not dict(presets.models_for("fsr4"))["anf"]
assert not dict(presets.models_for("static", anf=True))["anf"], "the RE4 interceptor has no ANF"
assert not dict(presets.models_for("fsr4", vulkan=True, anf=True))["anf"], "no ANF in a native Vulkan game"
assert [m for m, _ in presets.models_for("fsr4")] == ["fsr4", "fsr4fast", "anf", "fsr3", "dittosr", "sgsr2"]
READY = {"layer_found": True, "service": {"installed": True}}
assert presets.live_wanted("anf", READY) == {"model": "anf", "npu": 2, "npuexact": 1}
assert presets.live_wanted("anf", None) == {"model": "anf", "npu": None, "npuexact": None}
assert presets.live_wanted("sgsr2", READY) == {"model": "sgsr2", "npu": None, "npuexact": None}
assert presets.plan("fsr4", {"model": "anf", "npu": 2}) == {"kind": "fsr4", "model": "anf", "npu": True}
# FSR 4's "npu_runs" and graph hints stay FSR 4's: ANF is no FSR 4
assert not presets.npu_runs({"model": "anf"}, {"npu": {"state": "on"}}, READY, "1920x1080")

# ---- the DLL key and the chip
assert sgsr3.dx11_upscaler("anf") == "sgsr3_12" and sgsr3.VULKAN_UPSCALER["anf"] == "sgsr2"
assert device.policy({"soc": "SM8550"}, ())["anf"] is True
assert device.policy({"soc": "SM8650"}, ())["anf"] is False, "measured on the SM8550 only"
assert not device.policy({"soc": "SM8750"}, ()).get("anf") and not device.policy({"soc": "SM8250"}, ()).get("anf")
assert device.state({"soc": "SM8550"}, device.policy({"soc": "SM8550"}, ()))["anf"] is True

# ---- the NPU service: the package's families
HDR = "Num       RefCount Protocol Flags    Type St Inode Path\n"
LINE = "0000000000000000: 00000002 00000000 00010000 0005 01 12345 @%s\n"


def service(sockets, files):
    d = tempfile.mkdtemp(dir=tmp)
    proc, graphs = os.path.join(d, "unix"), os.path.join(d, "graphs")
    os.makedirs(graphs)
    for f in files:
        open(os.path.join(graphs, f), "w").close()
    with open(proc, "w") as fh:
        fh.write(HDR + "".join(LINE % s for s in sockets))
    return npu.Npu(d, d, d, log=lambda m: None, proc_net_unix=proc, systemctl=lambda u: "inactive", graph_dir=graphs).service(fresh=True)


p17 = service([npu.K1H_SOCKET], ["nf4_k1h.bin", "nf4_k1h_452.bin"])
assert p17["installed"] and p17["anf_graphs"] == [] and p17["anf_running"] == [], "package 17: no ANF"
p18 = service([npu.K1H_SOCKET, "fsr4npud_anf540s32"], ["nf4_k1h.bin", "nf4_k1h_452.bin", "anf540s32.bin", "anf360s32.bin"])
assert p18["anf_graphs"] == ["anf360s32", "anf540s32"] and p18["anf_running"] == ["anf540s32"], p18
assert npu.ANF_FAMILIES == ("anf360s32", "anf480s32", "anf540s32", "anf648s32", "anf720s32")

# ---- the plugin: hidden unless every condition holds (fake decky module, no __init__ run)
dk = types.ModuleType("decky")
dk.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
dk.DECKY_USER_HOME = os.path.join(tmp, "home")
dk.DECKY_PLUGIN_SETTINGS_DIR = os.path.join(tmp, "ps")
sys.modules["decky"] = dk
sys.path.insert(0, ROOT)
import main  # noqa: E402

P = main.Plugin()
chip = device.policy({"soc": "SM8550"}, ())
npu_ok = {"layer_found": True, "service": {"installed": True, "anf_graphs": ["anf540s32"]}}
s3 = {"fsr4_available": True, "fsr3_available": True, "dittosr_available": True, "anf_available": True, "npu": npu_ok}
P.chip = chip
assert dict(P._models("fsr4", s3))["anf"] is True
assert [m for m, ok in P._models("fsr4", s3) if ok][:3] == ["fsr4", "fsr4fast", "anf"]
for name, patch in (("the DLL has no ANF", {"anf_available": False}), ("package 17", {"npu": {**npu_ok, "service": {"installed": True, "anf_graphs": []}}}),
                    ("no layer", {"npu": {**npu_ok, "layer_found": False}}), ("no NPU part", {"npu": None})):
    assert dict(P._models("fsr4", {**s3, **patch}))["anf"] is False, name
for soc in ("SM8650", "SM8750"):
    P.chip = device.policy({"soc": soc}, ())
    assert dict(P._models("fsr4", s3))["anf"] is False, soc
P.chip = chip
assert dict(P._models("static", s3))["anf"] is False
assert dict(P._models("fsr4", s3, vulkan=True))["anf"] is False
# FSR 4 stays the default model; ANF is never chosen for a game by "auto"
assert P._default_model("fsr4", s3, npu_ok) == "fsr4"

# ---- the fork's log: ANF ran (the last live model is anf and its init line is in the run)
LOG = ("Init SGSR3 2.0: render 960x540 -> 1920x1080\nSGSR3: Vela live model anf\nInit Vela ANF: ready, render 960x540 -> 1920x1080, family anf540s32\n")
path = os.path.join(tmp, "OptiScaler.log")
open(path, "w").write(LOG)
src = open(os.path.join(ROOT, "py_modules", "sgsr3.py"), encoding="utf-8").read()
assert re.search(r'want\[-1\] == "anf" and "Vela ANF: ready" in seg', src)
r = sgsr3.log_summary(path)
assert r["mode"] == "anf", r["mode"]

# ---- the panel: name, note, status
ts = open(os.path.join(ROOT, "src", "strings.ts"), encoding="utf-8").read()
en = open(os.path.join(ROOT, "src", "i18n", "en.json"), encoding="utf-8").read()
st = open(os.path.join(ROOT, "src", "status.ts"), encoding="utf-8").read()
assert 'anf: "ANF"' in ts and '"model.note.anf": "' in en and 'L.mode === "anf"' in st
print("test_anf OK")
