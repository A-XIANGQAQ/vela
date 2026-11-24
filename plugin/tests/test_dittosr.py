"""DITTO (0.8.23, owner 2026-11-22 «Дито добавляй в основу»): DittoSR 0.5.4 is the live model "dittosr" wherever FSR 3 is offered
(DX12 and DX11 OptiScaler games; not the RE4 interceptor). The kit carries dittosr/amd_fidelityfx_upscaler_dx12.dll and
dittosr/dittosr.ini next to OptiScaler when FSR 3 can be a model; launch.json lists them, the repair puts them back, Off takes them
(and the folder) out again. The log: DittoSR runs, or "dittosr cannot run ... instead" -> FSR 3, a fallback the panel and the
report name. The plugin ships the DLL (md5 = DittoSR's release build), its ini with the defaults and DittoSR's licence.
A fresh temp folder per run (left in the system temp dir, nothing is deleted).
Run: python -I tests/test_dittosr.py"""
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import launchopts  # noqa: E402
import live  # noqa: E402
import presets  # noqa: E402
import report  # noqa: E402
import sgsr3  # noqa: E402

checks = 0


def ok(cond, what):
    global checks
    checks += 1
    assert cond, what


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


tmp = tempfile.mkdtemp(prefix="vela-ditto-")

# ---- 1. the model: live key, presets, panel order, DX11 upscaler
ok("dittosr" in presets.MODELS and "dittosr" in live.MODELS, presets.MODELS)
ok(presets.MODELS.index("dittosr") == presets.MODELS.index("fsr3") + 1, "DittoSR right after FSR 3")
ok(live.clean({"model": "dittosr"}) == {"model": "dittosr"}, "live file takes model=dittosr")
ok(live._tmp_line("model=dittosr") == "model=dittosr", "a /tmp line model=dittosr stays")
ok(dict(presets.models_for("fsr4", True, True, True))["dittosr"] is True, "offered with FSR 3")
ok(dict(presets.models_for("fsr4", True, False, True))["dittosr"] is False, "no FSR 3, no DittoSR (same FFX backend)")
ok(dict(presets.models_for("fsr4", True, True, False))["dittosr"] is False, "not in the build: greyed")
ok("dittosr" not in dict(presets.models_for("static", True, True, True)), "the RE4 interceptor has no DittoSR")
ok(sgsr3.dx11_upscaler("dittosr") == "ffx_12", "DX11: the FFX backend over the bridge, as FSR 3")
ok(sgsr3.ini_values({"live_model": "dittosr"})[("Upscalers", "Dx11Upscaler")] == "ffx_12", "DX11 ini")
ok(presets.plan("fsr4", {"model": "dittosr"})["model"] == "dittosr", "a picked DittoSR stays")
ok(presets.default_model("fsr4", True, None) != "dittosr", "not the default model (no owner word on that)")

# ---- 2. the kit: files when FSR 3 can be a model, launch.json, repair, Off
home = os.path.join(tmp, "home")
assets = os.path.join(tmp, "assets", "sgsr3")
for a in ("OptiScaler.dll", "OptiPatcher.asi", sgsr3.FFX_DLL):
    write(os.path.join(assets, a), b"asset " + a.encode())
shutil.copy(os.path.join(ROOT, "assets", "sgsr3", "OptiScaler.ini"), assets)
shutil.copy(os.path.join(ROOT, "assets", "sgsr3", "VERSION.json"), assets)
settings = os.path.join(tmp, "settings")
lib = os.path.join(home, ".local", "share", "Steam")
write(os.path.join(lib, "steamapps", "libraryfolders.vdf"), '"libraryfolders" { "0" { "path" "%s" } }' % lib.replace("\\", "/"))
for appid, name in (("1000", "Kit Game"), ("2000", "Copy Game")):
    write(os.path.join(lib, "steamapps", f"appmanifest_{appid}.acf"),
          '"AppState" { "appid" "%s" "name" "%s" "installdir" "%s" }' % (appid, name, name))
    g = os.path.join(lib, "steamapps", "common", name)
    write(os.path.join(g, "Game.exe"), b"x" * (3 << 20))
    write(os.path.join(g, "nvngx_dlss.dll"), b"dlss")
logs = []
s3 = sgsr3.Sgsr3(assets, home, settings, log=logs.append)
s3.probe = lambda exe, w, scratch: (True, "")
ok(s3.version.get("dittosr_live") is True and s3.version.get("fsr3_live") is True, "VERSION.json: the fork loads DittoSR")
ok(not s3.dittosr_available(), "no DittoSR files in assets: not available")
st0 = s3.status("1000")
ok(st0["dittosr_available"] is False and st0["fsr3_available"] is True, st0["dittosr_available"])
ok(not set(sgsr3.DITTO_FILES) & s3._kit_wanted(s3.kit("1000"), {}, {"files": {}}), "nothing to copy without the files")
# the plugin carries them (the real ones from assets/sgsr3/dittosr)
for rel in sgsr3.DITTO_FILES:
    write(os.path.join(assets, *rel.split("/")), open(os.path.join(ROOT, "assets", "sgsr3", *rel.split("/")), "rb").read())
s3.md5_cache.clear()
ok(s3.dittosr_available() and s3.status("1000")["dittosr_available"], "available with both files + fsr3")
s3.version = {**s3.version, "dittosr_live": False}
ok(not s3.dittosr_available(), "a fork without DittoSR: not offered")
s3.version = {**s3.version, "dittosr_live": True}
s3.version = {**s3.version, "fsr3_live": False}
ok(not s3.dittosr_available(), "FSR 3 not a model: no DittoSR")
s3.version = {**s3.version, "fsr3_live": True}

# overlay kit
s3.new_mode = "launcher"
s3.enable("1000", {}, launchopts.CMD)
files = s3.kit_files("1000")
lj_path = os.path.join(s3.kit_root("1000"), "launch.json")
lj = json.load(open(lj_path, encoding="utf-8"))
for rel in sgsr3.DITTO_FILES:
    ok(rel in lj["files"], (rel, lj["files"]))
    ok(md5(os.path.join(files, *rel.split("/"))) == md5(os.path.join(assets, *rel.split("/"))), rel)
ok(s3.kit_missing("1000") == [], s3.kit_missing("1000"))
# the repair: a changed / missing DittoSR file goes back, the list stays whole
os.remove(os.path.join(files, "dittosr", "dittosr.ini"))
write(os.path.join(files, "dittosr", "amd_fidelityfx_upscaler_dx12.dll"), b"someone else's")
logs.clear()
ok(s3.refresh_dll("1000") is True, "repair wrote")
for rel in sgsr3.DITTO_FILES:
    ok(md5(os.path.join(files, *rel.split("/"))) == md5(os.path.join(assets, *rel.split("/"))), ("repaired", rel))
ok(any("kit repaired" in x and sgsr3.DITTO_DLL in x for x in logs), logs)
ok(s3.refresh_dll("1000") is False, "a whole kit: nothing to write")
# the kit's launch.json lists the files even when an older Vela wrote it without them
lj = json.load(open(lj_path, encoding="utf-8"))
lj["files"] = [f for f in lj["files"] if not f.startswith("dittosr/")]
write(lj_path, json.dumps(lj, indent=2))
ok(s3.refresh_dll("1000") is True and set(sgsr3.DITTO_FILES) <= set(json.load(open(lj_path, encoding="utf-8"))["files"]),
   "list mended")

# copy setup: the files go next to the exe (OptiScaler's own folder), Off takes them and the folder out
s3.new_mode = "copy"
s3.enable("2000", {}, launchopts.CMD)
g = os.path.join(lib, "steamapps", "common", "Copy Game")
for rel in sgsr3.DITTO_FILES:
    ok(os.path.isfile(os.path.join(g, *rel.split("/"))), ("copy setup", rel))
st = s3.read_state(g)
ok("dittosr" in st["dirs_created"], st["dirs_created"])
s3.disable("2000", launchopts.CMD)
ok(not os.path.exists(os.path.join(g, "dittosr")), "Off: dittosr/ gone again")
# a copy setup an older Vela made (dirs_created without dittosr/): the repair records the folder it creates
s3.enable("2000", {}, launchopts.CMD)
st = s3.read_state(g)
for rel in sgsr3.DITTO_FILES:
    os.remove(os.path.join(g, *rel.split("/")))
os.rmdir(os.path.join(g, "dittosr"))
st["dirs_created"] = [r for r in st["dirs_created"] if r != "dittosr"]
s3._write_state(g, st)
ok(s3.refresh_dll("2000") is True and "dittosr" in s3.read_state(g)["dirs_created"], "repair records dittosr/")
s3.disable("2000", launchopts.CMD)
ok(not os.path.exists(os.path.join(g, "dittosr")), "Off after the repair: dittosr/ gone")

# ---- 3. the log: DittoSR runs / falls back to FSR 3
HEAD = "Initializing SGSR3 960x540 -> 1920x1080\nSGSR3: Vela live model fsr4\n"
SWITCH = "Vela live: handle 1000003 model dittosr -> DittoSR (the game draws 960x540)\n"
VERS = "Vela: FFX upscaler versions: 0=0.5.4, 1=3.1.4\n"


def summary(text, name="OptiScaler.log"):
    p = os.path.join(tmp, name)
    write(p, text)
    return sgsr3.log_summary(p)


# PANELFIX: the lines of real runs (b53122ed DittoSR log; FSR 3 lines of the KIT2 logs), not an invented "(0.5.4)"
REAL_DITTO = "[12:34:52.299028] [I] InitFFX Vela: FFX upscaler index 0 (DittoSR (FSR 3.1.5 API) 0.5.4) for DittoSR\n"
REAL_FSR3 = "[12:34:52.299028] [I] InitFFX Vela: FFX upscaler index 0 (3.1.4) for FSR 3\n"
for label, line in (("DittoSR line", REAL_DITTO), ("FSR 3 line (a fork naming the backend only)", REAL_FSR3)):
    o = summary(HEAD + SWITCH + VERS + line)
    ok(o["mode"] == "dittosr" and o["sgsr3"] and o["dittosr_fallback"] is None and o["render"] == "960x540", (label, o))
    ok(o["fsr3_fallback"] is None, o)
down = SWITCH + "Vela: can't load Z:\\g\\dittosr\\amd_fidelityfx_upscaler_dx12.dll\n" \
    "Vela live: dittosr cannot run in this game; FSR 3 instead\n" + "Vela: FFX upscaler index 0 (3.1.4) for FSR 3\n"
o = summary(HEAD + down)
ok(o["mode"] == "fsr3" and o["dittosr_fallback"] == "failed" and o["fsr3_version"] == "3.1.4" and o["sgsr3"], o)
o = summary(HEAD + SWITCH)  # asked, nothing ran yet / backend failed
ok(o["mode"] != "dittosr" and o["dittosr_fallback"] == "failed", o)
o = summary(HEAD + SWITCH + "Vela live: handle 1000003 model fsr3 -> FSR 3.1.4\nVela: FFX upscaler index 0 (3.1.4) for FSR 3\n")
ok(o["mode"] == "fsr3" and o["dittosr_fallback"] is None and o["fsr3_fallback"] is None, "a later pick of FSR 3 is no fallback")
ok(summary(HEAD)["dittosr_fallback"] is None, "never asked")

# the report names the fallback
src = open(os.path.join(ROOT, "py_modules", "report.py"), encoding="utf-8").read()
ok('"dittosr_fallback": bool(s.get("dittosr_fallback"))' in src and '"fsr3_fallback", "dittosr_fallback"' in src, "report why")
ok("sgsr3/dittosr/amd_fidelityfx_upscaler_dx12.dll" in src, "report md5 of the shipped DittoSR DLL")

# ---- 4. the panel: name, note, fallback card, report reason
en = json.load(open(os.path.join(ROOT, "src", "i18n", "en.json"), encoding="utf-8"))
ts = open(os.path.join(ROOT, "src", "strings.ts"), encoding="utf-8").read()
status_ts = open(os.path.join(ROOT, "src", "status.ts"), encoding="utf-8").read()
ok('dittosr: "DittoSR"' in ts and '"dittosr"' in re.search(r"MODEL_IDS: ModelId\[\] = \[[^\]]*\]", ts).group(0), "panel card")
note = en["model.note.dittosr"]
ok(len(note) <= 50 and "No neural network" in note and not re.search(r"[А-яЁё]", note), note)
ok(en["rp.why.dittosr_fallback"] == "DittoSR did not start" and en["st.dittosrFailedSub"], "fallback texts")
ok("L.dittosr_fallback" in status_ts and "st.dittosrFailedSub" in status_ts, "status card reads the fallback")

# ---- 5. what the plugin ships
pa = os.path.join(ROOT, "assets", "sgsr3", "dittosr")
ok(md5(os.path.join(pa, "amd_fidelityfx_upscaler_dx12.dll")) == "a98003a496aa3105c5f9bebf6588e0cc", "DittoSR 0.5.4 release DLL")
ini = open(os.path.join(pa, "dittosr.ini"), encoding="utf-8").read()
sec = {}
cur = None
for line in ini.splitlines():
    line = line.strip()
    if line.startswith("["):
        cur = line.strip("[]")
    elif "=" in line and not line.startswith(";"):
        k, _, v = line.partition("=")
        sec[(cur, k.strip())] = v.strip()
ok(sec[("dittosr", "enabled")] == "1" and sec[("debug", "enabled")] == "0" and sec[("debug", "indicator")] == "0", sec)
ok(sec[("dittosr", "version")] == "0.5.4", sec)
lic = os.path.join(ROOT, "assets", "licenses")
ok(open(os.path.join(lic, "dittosr", "DittoSR-MIT.txt"), encoding="utf-8").read().startswith("MIT License"), "licence")
ok(os.path.isfile(os.path.join(lic, "dittosr", "THIRD-PARTY-NOTICES.md")), "DittoSR's notice")
readme = open(os.path.join(lic, "README.txt"), encoding="utf-8").read()
ok("upscale/assets/sgsr3/dittosr/amd_fidelityfx_upscaler_dx12.dll" in readme and "dittosr/DittoSR-MIT.txt" in readme, "README")
for f in os.listdir(os.path.join(lic, "dittosr")):
    ok(f"dittosr/{f}" in readme, ("README names", f))
bz = open(os.path.join(ROOT, "build_zip.py"), encoding="utf-8").read()
ok('"assets/sgsr3/dittosr/amd_fidelityfx_upscaler_dx12.dll", "assets/sgsr3/dittosr/dittosr.ini"' in bz, "build_zip requires them")
gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
ok("assets/sgsr3/dittosr/*.dll" in gi, "the binary is not committed, as the other DLLs")
print(f"test_dittosr: {checks} checks, 0 failed")
