"""DX11FSR (0.8.23, fork patch 0027): a DirectX 11 game gets the same models as a DirectX 12 one (FSR 4, FSR 4 Fast, FSR 3,
SGSR 2). OptiScaler's Dx11Upscaler follows the live model (sgsr3_12 / ffx_12 / sgsr2_12, the fork switches at run time too),
the verdict of a DX11 game is kind fsr4 with dx11_only, and no panel text says DX11 = SGSR 2 only.
Run: python -I tests/test_dx11fsr.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import gamecheck  # noqa: E402
import presets  # noqa: E402
import sgsr3  # noqa: E402

DX11 = ("Upscalers", "Dx11Upscaler")
DX12 = ("Upscalers", "Dx12Upscaler")

# ---- 1. the ini per model in DX11 (and DX12 stays sgsr3: the fork's DX12 live switch picks the backend)
want = {"fsr4": "sgsr3_12", "fsr4fast": "sgsr3_12", "fsr3": "ffx_12", "sgsr2": "sgsr2_12",
        "fsr4lite": "sgsr3_12", None: "sgsr3_12", "ai": "sgsr3_12"}
for model, up in want.items():
    v = sgsr3.ini_values({"live_model": model})
    assert v[DX11] == up, (model, v[DX11])
    assert v[DX12] == "sgsr3", (model, v[DX12])
assert set(sgsr3.DX11_UPSCALER) == set(presets.MODELS), "every panel model has a DX11 upscaler"
tpl = "[Upscalers]\nDx12Upscaler=auto\nDx11Upscaler=auto\n"
assert "Dx11Upscaler=ffx_12" in sgsr3.make_ini(tpl, {"live_model": "fsr3"})
assert "Dx11Upscaler=sgsr2_12" in sgsr3.make_ini(tpl, {"live_model": "sgsr2"})

# the live file the panel writes names the model the ini takes
d = tempfile.mkdtemp(prefix="dx11fsr_")
with open(os.path.join(d, "123.ini"), "w", encoding="utf-8") as f:
    f.write("model=fsr4fast\nnpu=2\n")
assert sgsr3.live_model("123", d) == "fsr4fast" and sgsr3.live_model("999", d) is None
assert sgsr3.ini_values({"live_model": sgsr3.live_model("123", d)})[DX11] == "sgsr3_12"
src = open(os.path.join(ROOT, "py_modules", "sgsr3.py"), encoding="utf-8").read()
assert '"live_model": live_model(sc["appid"])' in src, "_ini_text passes the live model"

# ---- 2. the verdict and the model list for a DX11 game
base = {"install_dir": "/g", "exe_dir": "/g/bin", "exe": "g.exe", "inputs": ["DLSS"], "anticheat": []}
v11 = gamecheck.verdict(base, {"dx11"})
assert v11["kind"] == "fsr4" and v11["dx11_only"], v11
v12 = gamecheck.verdict(base, {"dx12", "dx11"})
assert v12["kind"] == "fsr4" and not v12["dx11_only"], v12
assert not gamecheck.verdict(base, set())["dx11_only"], "nothing read: DX12 assumed"
assert presets.models_for(v11["kind"]) == presets.models_for(v12["kind"]) == \
    [("fsr4", True), ("fsr4fast", True), ("fsr3", True), ("dittosr", True), ("sgsr2", True)]  # DITTO: + DittoSR
NPU_ON = {"layer_found": True, "service": {"installed": True}}
assert presets.default_model(v11["kind"], True, NPU_ON) == "fsr4"
assert presets.default_model(v11["kind"], True, None) == "fsr4fast"
for m in presets.MODELS:  # the picked model stays (0.8.22 forced SGSR 2 for DX11)
    assert presets.plan(v11["kind"], {"model": m})["model"] == m
assert "sgsr2" not in presets.KINDS
# a DX11 game that ran in DX11 is no reason to switch to DX12; a DX12 game that ran in DX11 still is
log11 = {"sgsr3": False, "api": "dx11"}
assert presets.hint({"kind": "fsr4", "model": "sgsr2"}, {"log": log11, "dx11_only": True}) is None
assert presets.hint({"kind": "fsr4", "model": "fsr4"}, {"log": log11, "dx11_only": True}) != {"id": "dx11_run"}
assert presets.hint({"kind": "fsr4", "model": "fsr4"}, {"log": log11}) == {"id": "dx11_run"}
main_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
assert 'if v.get("kind") == "sgsr2":' in main_src, "a 0.8.22 verdict (kind sgsr2) becomes kind fsr4"
assert '"dx11_only": bool(v.get("dx11_only"))' in main_src

# ---- 3. the texts: no "DX11 = SGSR 2 only", generic
en = json.load(open(os.path.join(ROOT, "src", "i18n", "en.json"), encoding="utf-8"))
assert "dx11.only" not in en
bad = [k for k, s in en.items() if "only SGSR 2" in s or "need DirectX 12" in s or "needs DirectX 12" in s]
assert not bad, bad
assert en["st.dx11"] == "The game ran in DirectX 11 and did not use its upscaler there.", en["st.dx11"]
tsx = open(os.path.join(ROOT, "src", "index.tsx"), encoding="utf-8").read()
st_ts = open(os.path.join(ROOT, "src", "status.ts"), encoding="utf-8").read()
assert 'kind === "sgsr2"' not in tsx and 'c.kind !== "sgsr2"' not in st_ts and "!c.dx11_only" in st_ts
print("test_dx11fsr OK")
