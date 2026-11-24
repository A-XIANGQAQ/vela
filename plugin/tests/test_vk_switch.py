"""0.9.0: a game that names Direct3D AND Vulkan (Unity, Unreal) is Vulkan only when it will run Vulkan: the launch options' last API
switch (-force-vulkan, -vulkan, --vulkan), else the last run's observed API, else Direct3D (Proton's default). No game list.
Run: python -I tests/test_vk_switch.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import gamecheck as g  # noqa: E402

SCAN = {"install_dir": "/g", "exe_dir": "/g", "exe": "Game.exe", "machine": "x64", "inputs": ["nvngx_dlss.dll"]}
UNITY = {"dx12", "dx11", "vulkan", "DLSS"}   # a Unity exe: both families named
DXONLY = {"dx12", "dx11", "DLSS"}
VKONLY = {"vulkan", "DLSS"}
GL = "VELA_NPU=1 ~/.local/share/vela/bin/vela-launch /usr/libexec/armada/armada-game-launch %command%"


def v(markers, lo=None, observed=None):
    return g.verdict(SCAN, markers, launch_options=lo, observed=observed)


# Unity exe with both markers, no switch: Direct3D, FSR 4 through DX12 as before (no vulkan key)
for lo in (None, "", GL, GL + " -force-d3d12"):
    r = v(UNITY, lo)
    assert r["kind"] == "fsr4" and not r.get("vulkan") and r["dx11_only"] is False, (lo, r)
# the same with a Vulkan switch: Vulkan
for sw in ("-force-vulkan", "-vulkan", "--vulkan", "--force-vulkan", "-FORCE-VULKAN"):
    r = v(UNITY, GL + " " + sw)
    assert r["kind"] == "fsr4" and r.get("vulkan") is True and r["dx11_only"] is False, (sw, r)
    assert v(UNITY, sw + " PROTON_LOG=1 %command%").get("vulkan") is True
# the last API switch wins; OpenGL is not Vulkan; a token that only contains the word is no switch
assert v(UNITY, GL + " -force-vulkan -force-d3d11").get("vulkan") is None
assert v(UNITY, GL + " -force-d3d11 -force-vulkan").get("vulkan") is True
assert v(UNITY, GL + " -force-glcore").get("vulkan") is None
assert v(UNITY, "MY_VULKAN_FLAG=1 VK_ICD=vulkan %command%").get("vulkan") is None
assert v(UNITY, "%command% -vulkanx").get("vulkan") is None
# the last run's observed API decides when the options have no API switch; a switch beats it
assert v(UNITY, GL, "vulkan").get("vulkan") is True
assert v(UNITY, GL, "dx11").get("vulkan") is None and v(UNITY, GL, "dx12").get("vulkan") is None
assert v(UNITY, GL + " -force-d3d12", "vulkan").get("vulkan") is None
assert v(UNITY, GL + " -force-vulkan", "dx11").get("vulkan") is True
# DX-only exe: never Vulkan, whatever the options say
for lo in (None, GL + " -force-vulkan", "-vulkan"):
    for ob in (None, "vulkan"):
        r = v(DXONLY, lo, ob)
        assert r["kind"] == "fsr4" and not r.get("vulkan"), (lo, ob, r)
# native Vulkan exe (no Direct3D named): Vulkan as before, with or without a switch
for lo in (None, GL, GL + " -force-vulkan"):
    r = v(VKONLY, lo)
    assert r["kind"] == "fsr4" and r.get("vulkan") is True, (lo, r)
# the switch parser alone
assert g.switch_api(None) is None and g.switch_api(5) is None and g.switch_api("") is None
assert g.switch_api("-dx12") == "d3d" and g.switch_api("-dx11 -vulkan") == "vulkan"
assert g.runs_vulkan(["dx12", "vulkan"], "-vulkan") and not g.runs_vulkan(["dx12"], "-vulkan") and g.runs_vulkan(["vulkan"])
# the gates before it are unchanged: anti-cheat and a 32-bit exe still block
assert g.verdict({**SCAN, "anticheat": "eac"}, UNITY, launch_options="-force-vulkan")["kind"] == "none"
assert g.verdict({**SCAN, "machine": "x86"}, UNITY, launch_options="-force-vulkan")["reason"] == "x86"

# ---- main.py wiring (source contract; the full backend path is bench/run_bench.py)
src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
assert "launch_options=launch_options, observed=(s3 or {}).get(\"api\")" in src
assert 'self._check(appid, gs["scan"], s3, ag, getattr(self, "_launch_seen", {}).get(appid))' in src, "panel state: Steam's options as read"
assert "steam_options if isinstance(steam_options, str) else launch_options)" in src, "apply_model: Steam's options reach the check"
print("test_vk_switch: OK")
