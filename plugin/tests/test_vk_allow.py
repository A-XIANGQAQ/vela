"""Vulkan GPU spoof for EVERY native Vulkan game (owner 2026-11-24; AUDIT/vkreach/NOTES.md): no per-game list. A native Vulkan game
with FSR 4 / FSR 4 Fast picked gets VELA_VK_SPOOF_GPU=nvidia + VELA_VK_FSR4=1 in its launch options; DX11/DX12 games never; any
other model or Vela off takes them out again, byte for byte. FSR 4 is offered in every native Vulkan game. Temp folders only.
Run: python -I tests/test_vk_allow.py"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import npu  # noqa: E402
import presets  # noqa: E402

SP, F4 = "VELA_VK_SPOOF_GPU=nvidia", "VELA_VK_FSR4=1"

# ---- no list any more; the pair and the models that need it
assert not hasattr(npu, "VK_ALLOW") and not hasattr(npu, "vk_allowed")
assert npu.VK_TOKENS == (SP, F4)
assert presets.VK_DLSS_MODELS == ("fsr4", "fsr4fast") == presets.FSR4_MODELS

# ---- the panel's models: FSR 4 is offered in every native Vulkan game (when the plugin carries it)
on = dict(presets.models_for("fsr4", True, True, True, vulkan=True))
assert on["fsr4"] and on["fsr4fast"] and on["fsr3"] and on["sgsr2"] and not on["anf"], on  # ANF never in a Vulkan game
assert dict(presets.models_for("fsr4", True, True, True, vulkan=True)) == dict(presets.models_for("fsr4", True, True, True, vulkan=True, anf=False))
assert not dict(presets.models_for("fsr4", False, True, True, vulkan=True))["fsr4"], "no FSR 4 build: still not offered"

tmp = tempfile.mkdtemp(prefix="vela_vk_")
assets = os.path.join(tmp, "assets", "npu")
os.makedirs(assets)
shutil.copy(os.path.join(ROOT, "assets", "npu", npu.MANIFEST), assets)
with open(os.path.join(assets, npu.LIB), "wb") as fh:
    fh.write(b"ELF")
n = npu.Npu(assets, os.path.join(tmp, "home"), os.path.join(tmp, "settings"), log=lambda *_: None)

# ---- tokens: the pair with vk, for any app id
for app in ("2940200", "1796470", "548430", "12345"):
    assert SP not in " ".join(n.tokens(app)) and n.tokens(app) == n.tokens(app, False)
    assert n.tokens(app, True)[-2:] == [SP, F4]

user = "PROTON_LOG=1 %command%"
for app in ("2940200", "548430", "99999999"):  # a listed-before game and two never listed
    lo0 = n.enable(app, user)                       # default: layer only
    assert SP not in lo0 and F4 not in lo0 and lo0.endswith(user), lo0
    assert n.read(app)["vk"] is False
    lo1 = n.enable(app, lo0, True)                  # FSR 4 picked in a native Vulkan game: the pair goes in, the player's stay
    assert SP in lo1.split() and F4 in lo1.split() and lo1.endswith(user), lo1
    assert n.read(app)["vk"] is True
    assert n.enable(app, lo1, True) == lo1 and n.enable(app, lo1) == lo1, "again / vk=None: unchanged"
    lo2 = n.enable(app, lo1, False)                 # another model: out again, byte for byte the layer-alone options
    assert lo2 == lo0 and n.read(app)["vk"] is False, (lo2, lo0)
    lo3 = n.enable(app, lo2, True)                  # Vela off: exactly the player's options
    assert n.disable(app, lo3) == user
    assert n.read(app) is None

# ---- the state remembers the pick when the options are rewritten without a flag (launch_fix and friends)
lo = n.enable("1796470", user, True)
assert n.enable("1796470", lo) == lo and SP in lo.split()
# ---- the player edits the options since: his tokens stay, ours are exchanged
edited = lo + " MY_VAR=1"
again = n.enable("1796470", edited, False)
assert SP not in again and "MY_VAR=1" in again and again.count("VELA_NPU=1") == 1, again

# ---- main.py wiring (source contract; the full backend path is bench/run_bench.py): the flag is Vulkan + model, never the app id
src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
assert 'vk = bool(v.get("vulkan")) and model in presets.VK_DLSS_MODELS' in src, "DX11/DX12 games (vulkan False) never get the pair"
assert 'out["vk_dlss"] = bool(v.get("vulkan")) and p["model"] in presets.VK_DLSS_MODELS' in src
assert "vk_allow" not in src and "vk_allowed" not in src
# ---- the panel line: one generic text, no game names
import json  # noqa: E402
line = json.load(open(os.path.join(ROOT, "src", "i18n", "en.json"), encoding="utf-8"))["vk.dlss"]
assert "DLSS" in line and "NVIDIA" in line and "Haste" not in line, line
print("test_vk_allow: OK")
