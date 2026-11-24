"""0.9.0 console defect (Haste Demo, 2026-11-24): in launcher / overlay kit mode apply_model edits the KIT's options
(s3.kit_options), and the API check read those, never Steam's `... %command% -force-vulkan`: no VK spoof tokens, no DLSS hint.
The real input of the device is used: Steam's options as the kit's pick read them. Run: python -I tests/test_vk_apply_path.py"""
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
tmp = tempfile.mkdtemp(prefix="vela_vkap_")
decky = types.ModuleType("decky")
decky.logger = types.SimpleNamespace(info=lambda *a: None, error=lambda *a: None, exception=lambda *a: None)
decky.DECKY_USER_HOME = os.path.join(tmp, "home")
decky.DECKY_PLUGIN_SETTINGS_DIR = os.path.join(tmp, "psettings")
sys.modules["decky"] = decky
sys.path.insert(0, ROOT)
import main  # noqa: E402
import gameutil  # noqa: E402

STEAM = "/usr/libexec/armada/armada-game-launch %command% -force-vulkan"   # as the console's pick read it (before)
KIT = "WINEDLLOVERRIDES=\"dxgi=n,b\" OPTI_SGSR3_STUDENT=1 %command%"       # the kit's own options (launch.json), no switch
gameutil.running_appids = lambda proc="/proc": set()


class Stop(Exception):
    pass


seen = {}
p = object.__new__(main.Plugin)
p.s3 = types.SimpleNamespace(kit_options=lambda appid: KIT, scan=lambda appid, fresh=False: {"install_dir": "/g"},
                             status=lambda appid: {"installed": False, "opts": {}})
p.ag = types.SimpleNamespace(status=lambda *a: {"installed": False})
p._kit_mode = lambda appid, model, lo: "launcher"
p._check = lambda appid, scan, s3, ag, lo=None: (seen.setdefault("lo", lo), (_ for _ in ()).throw(Stop()))[1]
try:
    p._apply_model("2940200", "fsr4", STEAM, "Default")
except Stop:
    pass
assert seen["lo"] == STEAM, seen   # the check reads Steam's options, not the kit's

# end to end through the real check: the verdict for Haste's markers is Vulkan with that string, Direct3D with the kit's
import gamecheck  # noqa: E402
SCAN = {"install_dir": "/g", "exe_dir": "/g", "exe": "Haste.exe", "machine": "x64", "inputs": ["nvngx_dlss.dll", "libxess.dll"]}
M = {"dx12", "dx11", "vulkan", "opengl"}   # gamecheck.json of the console: UnityPlayer.dll
assert gamecheck.verdict(SCAN, M, launch_options=STEAM).get("vulkan") is True
assert not gamecheck.verdict(SCAN, M, launch_options=KIT).get("vulkan")
# not launcher mode: unchanged
seen.clear()
p._kit_mode = lambda appid, model, lo: None
try:
    p._apply_model("2940200", "fsr4", STEAM, "Default")
except Stop:
    pass
assert seen["lo"] == STEAM, seen
print("test_vk_apply_path: OK")
