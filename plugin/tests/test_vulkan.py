"""VK1 (fork patch 0040): native Vulkan games get Vela's Vulkan live models (FSR 3, DittoSR, SGSR 2; FSR 4 via the DLSS path since 2026-11-24)."""
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import gamecheck  # noqa: E402
import presets  # noqa: E402
import sgsr3  # noqa: E402

# ---- 1. detection: Vulkan input markers in the exe, Vulkan input DLLs next to it
d = tempfile.mkdtemp(prefix="vk1_")
exe = os.path.join(d, "game.exe")
with open(exe, "wb") as f:
    f.write(b"MZ..vulkan-1.dll..vkCreateInstance..NVSDK_NGX_VULKAN_Init..xessVkCreateContext..")
found = gamecheck.read_markers(exe)
assert {"vulkan", "DLSS", "XeSS"} <= found and "dx12" not in found, found
assert sgsr3.INPUT_LIBS["amd_fidelityfx_vk.dll"] == "FSR 3.1"
assert sgsr3.INPUT_LIBS["ffx_fsr2_api_vk_x64.dll"] == "FSR 2"
assert sgsr3.INPUT_LIBS["nvngx_dlss.dll"] == "DLSS"  # RDR2 ships nvngx_dlss.dll next to RDR2.exe

# ---- 2. the verdict: Vulkan only + an input = kind fsr4, vulkan; D3D + Vulkan (RDR2, BG3) stays the D3D route
base = {"install_dir": "/g", "exe_dir": "/g", "exe": "g.exe", "inputs": ["DLSS"], "anticheat": []}
vk = gamecheck.verdict(base, {"vulkan"})
assert vk["kind"] == "fsr4" and vk["vulkan"] and not vk["dx11_only"], vk
mixed = gamecheck.verdict(base, {"dx12", "vulkan"})
assert mixed["kind"] == "fsr4" and not mixed.get("vulkan"), mixed
none = gamecheck.verdict({**base, "inputs": []}, {"vulkan"}, ag_assets=False)
assert none["reason"] == "vulkan_native" and none["kind"] == "none", none
assert gamecheck.verdict(base, {"opengl"})["kind"] == "none"

# ---- 3. the panel's models: everything but ANF offered (FSR 4 / FSR 4 Fast ride the game's DLSS path)
m = dict(presets.models_for("fsr4", True, True, True, vulkan=True))
assert m == {"fsr4": True, "fsr4fast": True, "anf": False, "fsr3": True, "dittosr": True, "sgsr2": True}, m  # ANF: never in a Vulkan game
assert not dict(presets.models_for("fsr4", True, True, True, vulkan=True, anf=True))["anf"]
assert [x for x, _ in presets.models_for("fsr4", vulkan=True)] == [x for x, _ in presets.models_for("fsr4")]
assert dict(presets.models_for("fsr4"))["fsr4"], "D3D games keep FSR 4"
src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
assert 'self._models(v["kind"], s3, bool(v.get("vulkan")), appid)' in src, "_model_state passes the Vulkan verdict"

# ---- 4. the ini: VulkanUpscaler follows the live model (no FSR 4 on Vulkan: SGSR 2); Vulkan spoofing only when asked
VK = ("Upscalers", "VulkanUpscaler")
for model, up in (("fsr3", "ffx"), ("dittosr", "ffx"), ("sgsr2", "sgsr2"), ("fsr4", "sgsr2"), ("fsr4fast", "sgsr2"), ("anf", "sgsr2"),
                  ("fsr4lite", "sgsr2"), (None, "sgsr2")):
    assert sgsr3.ini_values({"live_model": model})[VK] == up, (model, up)
assert set(sgsr3.VULKAN_UPSCALER) == set(presets.MODELS), "every panel model has a Vulkan upscaler"
tpl = "[Upscalers]\nDx12Upscaler=auto\nDx11Upscaler=auto\nVulkanUpscaler=auto\n[Spoofing]\nVulkan=auto\n"
assert "VulkanUpscaler=ffx" in sgsr3.make_ini(tpl, {"live_model": "fsr3"})
assert "Vulkan=false" in sgsr3.make_ini(tpl, {"live_model": "fsr3"})
assert "Vulkan=true" in sgsr3.make_ini(tpl, {"live_model": "fsr3", "vulkan_spoof": True})
ini = open(os.path.join(ROOT, "assets", "sgsr3", "OptiScaler.ini"), encoding="utf-8").read()
assert "\nVulkanUpscaler=auto" in ini.replace("\r\n", "\n"), "the template has the key make_ini sets"

# ---- 5. the kit carries AMD's FidelityFX SDK 1.1.4 Vulkan DLL with FSR 3 (and DittoSR's one DLL for both APIs)
assert sgsr3.FFXVK_DLL in sgsr3.INSTALLED and sgsr3.FFXVK_DLL in sgsr3.OPTIONAL
assert sgsr3.ASSET_OF[sgsr3.FFXVK_DLL] == "amd_fidelityfx_vk.dll"
s3src = open(os.path.join(ROOT, "py_modules", "sgsr3.py"), encoding="utf-8").read()
assert "rel == FFXVK_DLL and fsr3 and not kit[\"static\"] and self.ffxvk_available()" in s3src
assert sgsr3.DITTO_DLL == "dittosr/amd_fidelityfx_upscaler_dx12.dll"  # the fork's Vulkan loader takes it too

print("test_vulkan: OK")
