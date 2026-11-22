"""PANELWHY (0.8.23, players in the Armada Discord, Nov 2026): the status card says what really ran and why, in plain words.
1. FSR 4 fell back (SGSR 2 ran): the reason from the fork's log line (sgsr3.fsr4_why) -> one plain line; unknown -> "send a report".
   After the run the card keeps the last run's truth while an FSR 4 is still the pick.
2. A DirectX 11 game (kind sgsr2): one line why only SGSR 2 is offered; the DX12 start only when its own launch entries offer it;
   no "the game ran in DirectX 11 / needs DirectX 12" verdict and no dx11_run hint for such a game.
3. A run that never sent a frame through the game's upscaler: after WAIT_LONG_S the card says the game never asked for upscaling
   with the input Vela uses.
Part A: the log mapping (Python). Part B: src/status.ts + src/strings.ts compiled with TypeScript and run under Node.
Run: python -I tests/test_panelwhy.py   (part B needs node and typescript: ./node_modules or VELA_BENCH_NODE_MODULES)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "py_modules"))
import presets  # noqa: E402
import sgsr3  # noqa: E402

fails, checks = [], 0


def check(name, ok):
    global checks
    checks += 1
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


TMP = tempfile.mkdtemp(prefix="panelwhy_")


def summary(text, name="OptiScaler.log"):
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return sgsr3.log_summary(p)


# ---- A. the reason from the log ------------------------------------------------------------------------------------------
# the real lines of a player's run (report VELA-H78NGE, 2026-11-21): FSR 4 refused the depth format, SGSR 2 ran
s = sgsr3.log_summary(os.path.join(HERE, "fixtures", "panelwhy_depth_optiscaler.log"))
check(f"A1 real log: FSR 4 -> SGSR 2, reason depth_format ({s and s.get('fsr4_why')})",
      s and (s["mode"], s["fsr4_fallback"], s["sgsr2_instead"], s["fsr4_why"]) == ("sgsr2", "failed", "fsr4", "depth_format"))

HEAD = ("[I] InitInternal Initializing SGSR3 960x540 -> 1920x1080\n"
        "[I] CreatePipelines SGSR3: Vela live model fsr4\n")
INPUTS = "[I] EvaluateInternal SGSR3 inputs: color 960x540 fmt 10, mv 960x540 fmt 34 (sampled as 960x540), depth {d}, reactive false, " \
         "T&C false, output fmt 10 UAV true RT true, flags 0x33\n"
FAIL = "[E] EvaluateFsr4 SGSR3: FSR 4 VK frame failed; SGSR 2 from the next frame\n"
DOWN = ("[W] TryEvaluateOptiFeature Vela live: fsr4 cannot run in this game; SGSR 2 instead\n"
        "[I] TryEvaluateOptiFeature Vela live: handle 1 model fsr4 -> SGSR 2 (the game draws 960x540)\n"
        "[I] InitInternal Initializing SGSR2 960x540 -> 1920x1080\n")
UNSUP = "[E] Unsupported Vela FSR 4 VK: {w} format VkFormat 129 is not one the provider takes\n"
for what, why in (("depth", "depth_format"), ("motion vectors", "mv_format"), ("motion vectors (MV pass)", "mv_format"),
                  ("colour", "colour_format"), ("output", "output_format")):
    s = summary(HEAD + INPUTS.format(d=44) + UNSUP.format(w=what) + FAIL + DOWN)
    check(f"A2 '{what} format' -> {why} ({s['fsr4_why']})", (s["fsr4_fallback"], s["fsr4_why"]) == ("failed", why))
s = summary(HEAD + INPUTS.format(d=44) + "[E] Vela FSR 4 VK: ffxDispatch failed (6)\n" + FAIL + DOWN)
check(f"A3 ffx error 6 (output format) -> output_format ({s['fsr4_why']})", s["fsr4_why"] == "output_format")
s = summary(HEAD + INPUTS.format(d=-1) + FAIL + DOWN)
check(f"A4 the game gave no depth buffer -> no_depth ({s['fsr4_why']})", s["fsr4_why"] == "no_depth")
s = summary(HEAD + INPUTS.format(d=44) + "[W] Evaluate SGSR3: FSR 4 VK is not available here; SGSR 2 instead\n" + DOWN)
check(f"A5 no cause in the log -> unknown ({s['fsr4_why']})", (s["fsr4_fallback"], s["fsr4_why"]) == ("failed", "unknown"))
s = summary(HEAD + INPUTS.format(d=44) + "[I] Init Vela FSR 4 VK: FSR 4.1.1 VK INT8 context 960x540 -> 1920x1080 (hdr true)\n")
check(f"A6 FSR 4 runs: no reason ({s['fsr4_why']})", (s["mode"], s["fsr4_fallback"], s["fsr4_why"]) == ("fsr4", None, None))
# the last attempt is the one the card explains: an older refusal of the same run before another try does not count
s = summary(HEAD + UNSUP.format(w="depth") + FAIL + DOWN +
            "[I] TryEvaluateOptiFeature Vela live: handle 1 model sgsr2 -> SGSR 2\n"
            "[I] TryEvaluateOptiFeature Vela live: handle 1 model fsr4 -> FSR 4\n" + HEAD + UNSUP.format(w="motion vectors") + FAIL + DOWN)
check(f"A7 two tries: the last one's reason ({s['fsr4_why']})", s["fsr4_why"] == "mv_format")
check("A8 every reason id is in sgsr3.FSR4_WHY", set(sgsr3._FSR4_FMT_WHY.values()) | {"no_depth", "unknown"} == set(sgsr3.FSR4_WHY))
# the RE4 interceptor's log carries the field too (None when FSR 4 ran or was not picked)
p = os.path.join(TMP, "sgsr3intercept.log")
with open(p, "w", encoding="utf-8") as f:
    f.write("vela: live (x): model fsr4\nvela: FSR 4 VK is not available here\nvela: model fsr2 runs\n")
st = sgsr3.static_log_summary(p)
check(f"A9 interceptor: FSR 4 refused -> fsr4_why unknown ({st.get('fsr4_why')})", (st["fsr4_fallback"], st["fsr4_why"]) == ("failed", "unknown"))

# ---- A. DirectX 11: no DX12 verdict for a DirectX 11 game ------------------------------------------------------------------
dx11 = sgsr3.log_summary(os.path.join(HERE, "fixtures", "dx11_run_optiscaler.log"))
check("A10 a DX12 game that ran in DX11 still gets the dx11_run hint", presets.hint({"kind": "fsr4", "model": "fsr4"}, {"log": dx11}) == {"id": "dx11_run"})
check("A11 a DirectX 11 game (dx11_only) gets no dx11_run hint", presets.hint({"kind": "fsr4", "model": "fsr4"}, {"log": dx11, "dx11_only": True}) != {"id": "dx11_run"})
check("A12 a DirectX 11 game with a DX12 launch entry: pick it (dx12_choice)",
      presets.hint({"kind": "fsr4", "model": "fsr4"}, {"log": dx11, "dx12_choice": True, "dx11_only": True}) == {"id": "dx12_choice"})
main_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
check("A13 main.py gives the panel run_secs and dx12_choice", '"run_secs":' in main_src and 'out["dx12_choice"] = bool(v.get("dx11_only"))' in main_src)

# ---- B. the status card under Node -----------------------------------------------------------------------------------------
en = json.load(open(os.path.join(ROOT, "src", "i18n", "en.json"), encoding="utf-8"))
node = shutil.which("node")
NM = next((d for d in (os.path.join(ROOT, "node_modules"), os.environ.get("VELA_BENCH_NODE_MODULES") or "",
                       "C:/src/ext/vela7/appsFT3/decky-upscale/node_modules") if d and os.path.isdir(os.path.join(d, "typescript"))), None)
if not node or not NM:
    print("SKIPPED part B: node or typescript not found")
else:
    out = os.path.join(TMP, "b")
    os.makedirs(os.path.join(out, "i18n"))
    shutil.copy(os.path.join(ROOT, "src", "i18n", "en.json"), os.path.join(out, "i18n", "en.json"))
    comp = ("const ts=require(process.argv[1]);const fs=require('fs');for(const [src,dst] of [[process.argv[2],process.argv[3]],[process.argv[4],process.argv[5]]]){"
            "const r=ts.transpileModule(fs.readFileSync(src,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,"
            "esModuleInterop:true,resolveJsonModule:true},reportDiagnostics:true});"
            "if(r.diagnostics&&r.diagnostics.length){console.error(JSON.stringify(r.diagnostics.map(d=>d.messageText)));process.exit(2)}"
            "fs.writeFileSync(dst,r.outputText)}")
    c = subprocess.run([node, "-e", comp, os.path.join(NM, "typescript"), os.path.join(ROOT, "src", "status.ts"), os.path.join(out, "status.js"),
                        os.path.join(ROOT, "src", "strings.ts"), os.path.join(out, "strings.js")], capture_output=True, text=True)
    check(f"B0 status.ts and strings.ts compile ({(c.stderr or '').strip()[:120] or 'no diagnostics'})", c.returncode == 0)
    LOG0 = {"sgsr3": False, "render": None, "output": None, "mode": None}
    FB = {"sgsr3": True, "render": "960x540", "output": "1920x1080", "mode": "sgsr2", "fsr4_fallback": "failed", "sgsr2_instead": "fsr4"}

    def gs(log, running=True, kind="fsr4", model="fsr4", run_secs=None, static=False, dx11_only=False):
        return {"running": running, "check": {"kind": kind, "reason": None, "dx11_only": dx11_only}, "model": model, "run_secs": run_secs, "scan": {"inputs": ["DLSS"]},
                "sgsr3": {"installed": True, "intact": True, "log": log, "menu_hint": "choose DLSS", "static_fsr2": {} if static else None}}
    cases = {
        "fb_depth": gs({**FB, "fsr4_why": "depth_format"}),
        "fb_mv": gs({**FB, "fsr4_why": "mv_format"}),
        "fb_nodepth": gs({**FB, "fsr4_why": "no_depth"}),
        "fb_unknown": gs({**FB, "fsr4_why": "unknown"}),
        "fb_none": gs({**FB}),  # an older backend without the field
        "fb_fast": gs({**FB, "sgsr2_instead": "fsr4fast", "fsr4_why": "depth_format"}, model="fsr4fast"),
        "idle_fb": gs({**FB, "fsr4_why": "depth_format"}, running=False),
        "idle_fb_other_pick": gs({**FB, "fsr4_why": "depth_format"}, running=False, model="sgsr2"),
        "idle_ok": gs({**LOG0, "sgsr3": True, "mode": "fsr4"}, running=False),
        "dx11_game_waiting": gs({**LOG0, "api": "dx11"}, model="sgsr2", run_secs=30, dx11_only=True),
        "dx12_game_in_dx11": gs({**LOG0, "api": "dx11"}, run_secs=30),
        "wait_short": gs(LOG0, run_secs=60),
        "wait_long": gs(LOG0, run_secs=600),
        "wait_unknown_time": gs(LOG0, run_secs=None),
        "wait_long_static": gs(LOG0, run_secs=600, static=True),
        "dx11_game_long": gs({**LOG0, "api": "dx11"}, model="sgsr2", run_secs=600, dx11_only=True),
    }
    if c.returncode == 0:
        drv = os.path.join(out, "drv.js")
        with open(drv, "w", encoding="utf-8") as f:
            f.write("const s=require('./status.js');const cases=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));"
                    "const o={WAIT_LONG_S:s.WAIT_LONG_S};for(const k in cases)o[k]=s.statusOf(cases[k]);console.log(JSON.stringify(o))")
        cj = os.path.join(out, "cases.json")
        json.dump(cases, open(cj, "w", encoding="utf-8"))
        r = subprocess.run([node, drv, cj], capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=out)
        check(f"B1 the card runs under Node ({(r.stderr or '').strip()[:160] or 'ok'})", r.returncode == 0)
        R = json.loads(r.stdout) if r.returncode == 0 else {}
        lines = lambda k: [R.get(k, {}).get(f) for f in ("text", "sub", "why")]
        WAIT = R.get("WAIT_LONG_S")
        sys.stdout.reconfigure(errors="backslashreplace")  # a console without UTF-8 (cp1251) still prints the cards
        print("  cards:")
        for k in cases:
            print(f"    {k}: {lines(k)}")
        # 1. FSR 4 fell back, the reason in plain words
        check("B2 running: 'FSR 4 couldn't start in this game. SGSR 2 runs instead.' + the depth line",
              lines("fb_depth")[:2] == ["FSR 4 couldn't start in this game. SGSR 2 runs instead.",
                                         "Its depth buffer is in a format FSR 4 doesn't take."]
              and lines("fb_depth")[2] == "960×540 → 1920×1080")
        check("B3 motion vectors / no depth lines", lines("fb_mv")[1] == "Its motion vectors are in a format FSR 4 doesn't take."
              and lines("fb_nodepth")[1] == "It gave no depth buffer, and FSR 4 needs one.")
        check("B4 unknown reason and an older backend: send a report", lines("fb_unknown")[1] == lines("fb_none")[1] == "Send a report so we can look.")
        check("B5 FSR 4 Fast named as the one that fell back", lines("fb_fast")[0] == "FSR 4 Fast couldn't start in this game. SGSR 2 runs instead."
              and lines("fb_fast")[1] == "Its depth buffer is in a format FSR 4 Fast doesn't take.")
        check("B6 after the run: the last run's truth while FSR 4 is the pick",
              lines("idle_fb") == ["On · FSR 4", "Last run: FSR 4 couldn't start in this game. SGSR 2 ran instead.",
                                   "Its depth buffer is in a format FSR 4 doesn't take."])
        check("B7 after the run, another model picked: no old FSR 4 story", lines("idle_fb_other_pick") == ["On · SGSR 2", en["st.onIdleSub"], None])
        check("B8 after a good run: the usual idle card", lines("idle_ok") == ["On · FSR 4", en["st.onIdleSub"], None])
        # 2. DirectX 11
        check("B9 a DirectX 11 game before its first frame: waiting, no 'needs DirectX 12' verdict",
              R.get("dx11_game_waiting", {}).get("text") == "SGSR 2 · waiting for the first frame")
        check("B10 a DX12 game that ran in DX11: the plain DX11 line", lines("dx12_game_in_dx11")[0] == en["st.dx11"])
        # 3. waiting
        check("B11 a short wait: the usual waiting card", lines("wait_short")[0] == "FSR 4 · waiting for the first frame")
        check(f"B12 after {WAIT} s: the game never asked for upscaling",
              lines("wait_long") == ["The game never asked for upscaling with the input Vela uses.",
                                     "Try another upscaler in the game's settings (DLSS, FSR 3 or XeSS).", None])
        check("B13 no run time known: the usual waiting card", lines("wait_unknown_time")[0] == "FSR 4 · waiting for the first frame")
        check("B14 the game's own FSR 2 (interceptor): no 'try another upscaler'", lines("wait_long_static")[0] == "FSR 4 · waiting for the first frame")
        check("B15 a DirectX 11 game that never asked: the same long-wait card", lines("dx11_game_long")[0] == en["st.neverAsked"])
        check("B16 WAIT_LONG_S is 300 s", WAIT == 300)

# ---- the texts: generic, plain, the panel shows them
check("C1 DX11FSR: no 'DX11 = SGSR 2 only' text; no text says FSR 4 / FSR 3 need DirectX 12", "dx11.only" not in en and
      not [k for k, v in en.items() if "need DirectX 12" in v or "needs DirectX 12" in v or "only SGSR 2" in v])
check("C2 no 'Vela needs / works in DirectX 12' claim and no DirectX words in the speed line",
      not [k for k, v in en.items() if "Vela needs DirectX" in v or "Vela works in DirectX" in v] and "DirectX" not in en["speed.sgsr2"])
tsx = open(os.path.join(ROOT, "src", "index.tsx"), encoding="utf-8").read()
check("C3 the panel has no DX11-only line and shows the DX12 start for a DirectX 11 game", 'dx11.only' not in tsx and 'gs.check.dx11_only && gs.dx12_choice' in tsx
      and 'gs.dx12_choice' in tsx and 'from "./status"' in tsx)
print(f"test_panelwhy: {checks} checks, {len(fails)} failed" + (f": {fails}" if fails else ""))
raise SystemExit(1 if fails else 0)
