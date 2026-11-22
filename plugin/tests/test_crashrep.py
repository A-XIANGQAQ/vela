"""CRASHREP (0.8.23): games that die early on Snapdragon 8 Elite devices sent reports that end before any upscaler call and say
nothing about the cause. vela-launch now keeps the game's stderr (Proton / Wine: err: lines, "Unhandled exception ... in 64-bit
code", the backtrace with module names) per run in the kit folder (stderr-YYYYMMDD-HHMMSS.log, the run's start; last 10 runs,
each capped); the report attaches the reported run's file as game_stderr.txt (anonymised) and says in meta.json where the game
crashed (flags.crash_module / crash_addr / crash_ours, crash: one line).
Checked here: the copier (both the stderr Steam gave and the file get every byte; a failing file or a failing stderr leaves the
other working; the cap keeps the start and the newest output), the rotation (only stderr-*.log, the newest 10), the parser on
Wine's crash output formats (winedbg "Unhandled exception:" + Backtrace + Modules, ntdll's "wine: Unhandled ...", a missing DLL),
the report (the run's own file, anonymised, the summary, no file -> said, no error). POSIX only (fork + exec, skipped on Windows):
the real launcher end to end, the game's exit code and its death by signal unchanged, the start cost.
Fixtures: tests/fixtures/stderr_*.txt (written from Wine's documented crash output formats). Run: python -I tests/test_crashrep.py"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, "..", "py_modules")
sys.path.insert(0, PY)
import report  # noqa: E402
import vela_launch  # noqa: E402

checks = 0


def ok(cond, what):
    global checks
    checks += 1
    assert cond, what


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())


def fixture(name):
    with open(os.path.join(HERE, "fixtures", name), encoding="utf-8") as f:
        return f.read()


tmp = tempfile.mkdtemp(prefix="vela-crashrep-")


def run_pump(chunks, path, out=True, cap=vela_launch.STDERR_CAP, head=vela_launch.STDERR_HEAD):
    """The copier between two pipes (the game's stderr -> the parent's stderr + the file); returns what the parent got."""
    r, w = os.pipe()
    orr, ow = os.pipe()
    got = []
    reader = threading.Thread(target=lambda: got.append(b"".join(iter(lambda: os.read(orr, 1 << 16), b""))))
    reader.start()
    if not out:
        os.close(ow)
    t = threading.Thread(target=vela_launch.pump, args=(r, ow if out else 12345, path, cap, head))
    t.start()
    for c in chunks:
        os.write(w, c)
    os.close(w)
    t.join(20)
    ok(not t.is_alive(), "the copier ends when the game closes its stderr")
    if out:
        os.close(ow)
    reader.join(20)
    os.close(r)
    os.close(orr)
    return got[0] if got else b""


# 1) the copier: every byte to both, in order
crash = fixture("stderr_crash_dxgi.txt").encode()
p1 = os.path.join(tmp, "k1", "stderr-20261121-100000.log")
os.makedirs(os.path.dirname(p1))
chunks = [b"0124:err:module:x\n", crash[:500], crash[500:], b"tail line\n"]
seen = run_pump(chunks, p1)
want = b"".join(chunks)
ok(seen == want, "the parent's stderr gets every byte, unchanged")
ok(open(p1, "rb").read() == want, "the file gets the same bytes")
# a file that cannot be written: the game's stderr still reaches Steam
ok(run_pump([b"a\n", b"b\n"], os.path.join(tmp, "no", "such", "dir", "x.log")) == b"a\nb\n", "no file: forwarding goes on")
# a dead parent stderr: the file still gets everything (and the pipe is drained: the copier ends)
p2 = os.path.join(tmp, "k1", "stderr-20261121-100100.log")
run_pump([b"x" * 70000, b"end\n"], p2, out=False)
ok(open(p2, "rb").read() == b"x" * 70000 + b"end\n", "no parent stderr: the file still gets every byte")
# the cap: the start (Proton's first lines) + a cut line + the newest output, never over the cap
p3 = os.path.join(tmp, "k1", "stderr-20261121-100200.log")
lines = [f"{i:06d}:err:line {i} {'.' * 40}\n".encode() for i in range(20000)]
seen = run_pump([b"".join(lines[i:i + 50]) for i in range(0, len(lines), 50)], p3, cap=64 << 10, head=8 << 10)
ok(seen == b"".join(lines), "capped file: the parent still got all of it")
data = open(p3, "rb").read()
ok(len(data) <= 64 << 10, len(data))
ok(data.startswith(lines[0] + lines[1]) and data.endswith(lines[-1]), "the start and the newest lines stay")
ok(b"vela-launch: output cut here" in data and data.count(b"vela-launch: output cut here") == 1, "one cut line")
cut = data.split(b"\n")
ok(all(x == b"" or x.startswith(b"0") or x.startswith(b"... [vela-launch") for x in cut), "whole lines on both sides of the cut")

# 2) rotation: only stderr-*.log files, the newest STDERR_RUNS - 1 before a new run (the new file makes STDERR_RUNS)
kd = os.path.join(tmp, "kits", "2634950")
for i in range(12):
    write(os.path.join(kd, f"stderr-202611{10 + i:02d}-120000.log"), f"run {i}\n")
for other in ("launch.log", "launch.json", "stderr-notes.txt", "stderr-20261101-120000.log.1", "OptiScaler.log"):
    write(os.path.join(kd, other), "keep\n")
vela_launch.rotate_stderr(kd)
left = sorted(n for n in os.listdir(kd) if vela_launch.STDERR_RX.match(n))
ok(len(left) == vela_launch.STDERR_RUNS - 1 and left[0] == "stderr-20261113-120000.log" and left[-1] == "stderr-20261121-120000.log", left)
ok(all(os.path.isfile(os.path.join(kd, o)) for o in ("launch.log", "launch.json", "stderr-notes.txt", "stderr-20261101-120000.log.1",
                                                     "OptiScaler.log")), "no other file of the kit folder is touched")
ok(vela_launch.STDERR_RUNS == 10 and vela_launch.STDERR_CAP == 2 << 20, "10 runs per game, 2 MB per file")
ok(os.path.basename(vela_launch.stderr_path(kd, time.mktime((2026, 11, 21, 17, 29, 45, 0, 0, -1)))) == "stderr-20261121-172945.log",
   "named by the run's start, local time")

# 3) which games: a kit that is on; not a kit that is off, not a game without a kit
kits = os.path.join(tmp, "kits")
write(os.path.join(kits, "2634950", "launch.json"), json.dumps({"on": True}))
write(os.path.join(kits, "111", "launch.json"), json.dumps({"on": False}))
ok(vela_launch.stderr_dir({"SteamAppId": "2634950"}, kits) == os.path.join(kits, "2634950"), "kit on: kept")
ok(vela_launch.stderr_dir({"SteamAppId": "111"}, kits) is None, "kit off: nothing kept")
ok(vela_launch.stderr_dir({"SteamAppId": "222"}, kits) is None and vela_launch.stderr_dir({}, kits) is None, "no kit: nothing kept")
env = {"SteamAppId": "2634950", vela_launch.DONE: "1"}
ok(vela_launch.keep_stderr(env, kits) is False and vela_launch.TEE not in env, "the kit's second pass keeps nothing twice")
env = {"SteamAppId": "2634950", vela_launch.TEE: "1"}
ok(vela_launch.keep_stderr(env, kits) is False, "a second launcher pass keeps nothing twice")
if os.name != "posix":
    env = {"SteamAppId": "2634950"}
    ok(vela_launch.keep_stderr(env, kits) is False and vela_launch.TEE not in env, "no fork (Windows): nothing, fd 2 untouched")
src = open(os.path.join(PY, "vela_launch.py"), encoding="utf-8").read()
for var in ("WINEDEBUG", "PROTON_LOG", "DXVK_LOG", "VKD3D_DEBUG"):
    ok(f'"{var}' not in src and f"{var}=" not in src, f"the launcher sets no {var}")

# 4) the parser on Wine's crash output
c = report.crash_info(fixture("stderr_crash_dxgi.txt"), True)
ok(c["module"] == "dxgi" and c["offset"] == "0x123ab" and c["addr"] == "0x6fffff4123ab", c)
ok(c["kind"] == "page fault on read access to 0x10" and c["ours"] is True and "OptiScaler" in c["ours_what"], c)
ok(c["frames"][:2] == ["dxgi", "game-win64-shipping"], c["frames"])
ok(report.crash_summary(c) == "page fault on read access to 0x10 in dxgi (+0x123ab) at 0x6fffff4123ab: ours, OptiScaler, "
   "the dxgi.dll of Vela's kit", report.crash_summary(c))
ok(report.crash_info(fixture("stderr_crash_dxgi.txt"), False)["ours"] is False, "a run without the kit: dxgi.dll is Wine's own")
c = report.crash_info(fixture("stderr_crash_exe.txt"), True)
ok(c["module"] == "game" and c["addr"] == "0x141f3e07c" and c["ours"] is False and c["ours_in_stack"] is True, c)
ok(report.crash_summary(c).endswith(": not one of Vela's files"), report.crash_summary(c))
c = report.crash_info(fixture("stderr_crash_throw.txt"), True)  # a C++ throw: frame 0 in KERNELBASE, raised from the FSR provider
ok(c["module"] == "kernelbase" and c["app_module"] == "amd_fidelityfx_upscaler_vk" and c["ours"] is True, c)
ok(c["kind"].startswith("C++ exception") and "called from amd_fidelityfx_upscaler_vk" in report.crash_summary(c), c)
c = report.crash_info(fixture("stderr_nodll.txt"), True)
ok(c["kind"] == "missing DLL VCRUNTIME140_1.dll" and c["needed_by"] == "game.exe" and c["ours"] is False, c)
ok(report.crash_summary(c) == "missing DLL VCRUNTIME140_1.dll (needed by game.exe)", report.crash_summary(c))
c = report.crash_info(fixture("stderr_ntdll_only.txt"), True)  # no debugger ran: the address, no module
ok(c["kind"] == "illegal instruction" and c["addr"] == "0x6ffffe9a1000" and c["module"] is None and c["ours"] is None, c)
ok(report.crash_info(fixture("stderr_clean.txt"), True) == {} and report.crash_info("", True) == {}, "no crash: nothing")
ok(report.crash_info("wine: Call from 00006FFFFFC3A1E4 to unimplemented function api-ms-win-core-x-l1-1-0.dll.Foo, aborting\n",
                     True)["kind"] == "unimplemented function api-ms-win-core-x-l1-1-0.dll.Foo", "unimplemented function")
ok(report.module_stem("C:\\x\\DXGI.DLL") == "dxgi" and report.module_stem("libVkLayer_VELA_npu.so") == "libvklayer_vela_npu", "stems")
ok(report.ours_label("libVkLayer_VELA_npu.so") and report.ours_label("OptiPatcher.asi") and report.ours_label("amd_fidelityfx_dx12.dll")
   and report.ours_label("OptiScaler.dll") and not report.ours_label("game.exe") and not report.ours_label("ntdll"), "ours")
ok(report.crash_info("Segmentation fault (core dumped)\n")["kind"] == "Segmentation fault", "a Linux-side death")


# 5) the report: the reported run's stderr, anonymised, + the summary in meta.json
class S3:
    def __init__(self, root):
        self.root = root
        self.logs_dir = os.path.join(root, "settings", "sgsr3-logs")
        self.kits_dir = os.path.join(root, "settings", "kits")

    def kit_root(self, appid):
        return os.path.join(self.kits_dir, str(appid))

    def scan(self, appid):
        return {"exe_dir": os.path.join(self.root, "game"), "install_dir": os.path.join(self.root, "game")}

    def kit(self, appid):
        return {"log": "OptiScaler.log", "static": None}


APP = "2634950"
HOME = "/var/home/armada"
T = 1795022987.0
s3 = S3(tmp)
clock = [T + 600]
A = report.Anon(HOME, "armada", ("armada-host",), ("zrlogin",), ())
R = report.Reporter(os.path.join(tmp, "settings"), HOME, tmp, s3=s3, now=lambda: clock[0], anon=A, log=lambda *a: None,
                    runner=lambda *a, **k: (_ for _ in ()).throw(OSError("no tool")), proc=os.path.join(tmp, "proc"),
                    os_release_path=os.path.join(tmp, "none"), proc_osrelease=os.path.join(tmp, "none"))
R._write("run_starts.json", [{"appid": APP, "at": int(T - 900), "cmd": "vela-launch x", "vela": True, "end": int(T - 850)},
                             {"appid": APP, "at": int(T), "cmd": "~/.local/share/vela/bin/vela-launch x %command%", "vela": True,
                              "end": int(T + 71)}])
kroot = s3.kit_root(APP)


def stderr_at(t, text):
    write(vela_launch.stderr_path(kroot, t), text)


secret = ("vela-launch: kit on: /var/home/armada/.local/share/Steam/steamapps/common/Game/Binaries/Win64 (5 entries)\n"
          "0124:err:module:x L\"Z:\\var\\home\\armada\\.local\\share\\Steam\\game.exe\" user armada on armada-host, "
          "API_KEY=abc123secret steam 76561198000000001\n")
stderr_at(T - 902, "an earlier run\n" + fixture("stderr_crash_exe.txt"))
stderr_at(T - 2, secret + fixture("stderr_crash_dxgi.txt"))   # vela-launch started 2 s before the plugin saw the game
p = R.build(APP, "crash", "", "~/.local/share/vela/bin/vela-launch %command%", None, "Game")
z = zipfile.ZipFile(p["path"])
names = z.namelist()
ok("game_stderr.txt" in names and "game_stderr.txt" in [f["name"] for f in p["files"]], names)
se = z.read("game_stderr.txt").decode()
ok(se.startswith("# the game's stderr of the run that started ") and "stderr-" in se.splitlines()[0], se.splitlines()[0])
ok("Unhandled exception: page fault on read access" in se and "=>0 0x00006fffff4123ab in dxgi" in se, "the crash lines are there")
ok("an earlier run" not in se, "not another run's file")
for bad in ("/var/home/armada", "\\var\\home\\armada", "abc123secret", "76561198000000001", "armada-host", " armada "):
    ok(bad not in se, bad)
ok("~/.local/share/Steam" in se and "<redacted>" in se and "<user>" in se, "anonymised the same way as the rest")
meta = json.loads(z.read("meta.json"))
ok(meta["stderr"] == os.path.basename(vela_launch.stderr_path(kroot, T - 2)), meta["stderr"])
ok(meta["flags"]["crash_module"] == "dxgi" and meta["flags"]["crash_addr"] == "0x6fffff4123ab" and meta["flags"]["crash_ours"] is True,
   meta["flags"])
ok(meta["crash"].startswith("page fault on read access to 0x10 in dxgi (+0x123ab)") and "ours, OptiScaler" in meta["crash"], meta["crash"])
ok(meta["crash_detail"]["frames"][0] == "dxgi" and meta["crash_detail"]["ours_in_stack"] is True, meta["crash_detail"])
readme = z.read("README.txt").decode()
ok("game_stderr.txt" in readme and "crash:" in readme, "README lists the new file")

# the earlier run (a kept log's record) gets its own file: the game's exe crashed there, not ours
R._write("run_starts.json", [{"appid": APP, "at": int(T - 900), "cmd": "vela-launch x", "vela": True, "end": int(T - 850)}])
z.close()
clock[0] += 2  # a new report id
p = R.build(APP, "crash", "", None, None, "Game")
meta = json.loads(zipfile.ZipFile(p["path"]).read("meta.json"))
ok(meta["flags"]["crash_module"] == "game" and meta["flags"]["crash_ours"] is False and "an earlier run" in
   zipfile.ZipFile(p["path"]).read("game_stderr.txt").decode(), meta["flags"])

# a run with no stderr file (an older launcher, the kit off): no file, meta says so, no error
R._write("run_starts.json", [{"appid": APP, "at": int(T + 5000), "cmd": "x", "vela": True}])
clock[0] = T + 5100
p = R.build(APP, "crash", "", None, None, "Game")
z = zipfile.ZipFile(p["path"])
meta = json.loads(z.read("meta.json"))
ok("game_stderr.txt" not in z.namelist() and meta["stderr"].startswith("none:"), meta["stderr"])
ok(meta["flags"]["crash_module"] is None and meta["crash"] is None and not any("stderr" in e for e in meta["errors"]), meta)
# a run whose stderr says nothing about a crash: attached, "no crash"
stderr_at(T + 4998, fixture("stderr_clean.txt"))
clock[0] += 2  # a new report id
p = R.build(APP, "crash", "", None, None, "Game")
meta = json.loads(zipfile.ZipFile(p["path"]).read("meta.json"))
ok(meta["crash"] == "no crash in the game's stderr" and meta["flags"]["crash_ours"] is None, meta["crash"])
# no record at all: the newest file inside the run window (a running game: its newest)
R2 = report.Reporter(os.path.join(tmp, "s2"), HOME, tmp, s3=s3, now=lambda: clock[0], anon=A, log=lambda *a: None,
                     runner=lambda *a, **k: (_ for _ in ()).throw(OSError("no tool")), proc=os.path.join(tmp, "proc"))
ok(R2.stderr_file(APP, None, int(T - 60), int(T + 120)) == vela_launch.stderr_path(kroot, T - 2), "window match")
ok(R2.stderr_file(APP, None, 0, 1, running=True) == vela_launch.stderr_path(kroot, T + 4998), "running: the newest")
ok(R2.stderr_file("999", None, 0, 1 << 40) is None, "a game without files")

# 6) POSIX: the real launcher, end to end (fork + exec): the game is the process, its stderr reaches both
if hasattr(os, "fork"):
    import signal
    kits2 = os.path.join(tmp, "kits2")
    write(os.path.join(kits2, APP, "launch.json"), json.dumps({"on": True, "game_dir": "/nonexistent/game", "layer": "/nonexistent/l",
                                                               "work_dir": "/nonexistent/w"}))
    harness = os.path.join(tmp, "harness.py")
    write(harness, "import sys, time\nsys.path.insert(0, %r)\nimport vela_launch\nvela_launch.KITS = %r\n"
                   "t = time.perf_counter()\nvela_launch.keep_stderr()\nsys.stdout.write('%%.4f\\n' %% (time.perf_counter() - t))\n"
                   "sys.stdout.flush()\nvela_launch.main(sys.argv[1:])\n" % (PY, kits2))
    game = ("import os, sys, signal\nsys.stderr.write('0124:err:game says hello\\n' + 'x' * 200000 + '\\nlast\\n')\nsys.stderr.flush()\n"
            "print('stdout ' + os.environ.get('VELA_LAUNCH_STDERR', '-'), flush=True)\n"
            "if sys.argv[1] == 'segv':\n    os.kill(os.getpid(), signal.SIGSEGV)\nsys.exit(int(sys.argv[1]))\n")
    genv = {k: v for k, v in os.environ.items() if k not in (vela_launch.DONE, vela_launch.TEE)}
    genv["SteamAppId"] = APP
    for arg, code in (("7", 7), ("0", 0), ("segv", -signal.SIGSEGV)):
        before = set(os.listdir(os.path.join(kits2, APP)))
        r = subprocess.run([sys.executable, "-I", harness, sys.executable, "-I", "-c", game, arg], capture_output=True, env=genv,
                           timeout=60)
        ok(r.returncode == code, (arg, r.returncode, r.stderr[-400:]))
        ok(b"0124:err:game says hello\n" in r.stderr and r.stderr.rstrip().endswith(b"last") or arg == "segv" and b"last" in r.stderr,
           r.stderr[-200:])
        out = r.stdout.decode().split()
        ok(float(out[0]) < 0.25, f"the copier starts in {out[0]} s")
        ok("stdout 1" in r.stdout.decode(), "the game's stdout is untouched; the marker is set")
        new = sorted(set(os.listdir(os.path.join(kits2, APP))) - before)
        files = [n for n in new if vela_launch.STDERR_RX.match(n)]
        if not files:  # the same second as the run before: the same file (appended)
            files = sorted(n for n in os.listdir(os.path.join(kits2, APP)) if vela_launch.STDERR_RX.match(n))[-1:]
        kept = open(os.path.join(kits2, APP, files[-1]), "rb").read()
        ok(b"0124:err:game says hello\n" in kept and b"last\n" in kept, "the file has the game's stderr")
        ok(b"vela-launch: Vela left out of this launch" in kept, "and the launcher's own lines of the run")
        time.sleep(1.1)
    # a launch through two launcher passes (Steam's options + the Armada wrapper) keeps one file
    before = set(os.listdir(os.path.join(kits2, APP)))
    r = subprocess.run([sys.executable, "-I", harness, sys.executable, "-I", harness, sys.executable, "-I", "-c", game, "3"],
                       capture_output=True, env=genv, timeout=60)
    new = [n for n in set(os.listdir(os.path.join(kits2, APP))) - before if vela_launch.STDERR_RX.match(n)]
    ok(r.returncode == 3 and len(new) <= 1 and r.stderr.count(b"game says hello") == 1, (r.returncode, new))
    print("test_crashrep: POSIX end-to-end part ran")
else:
    print("test_crashrep: POSIX end-to-end part skipped (no fork on this platform)")
print(f"test_crashrep: {checks} checks, 0 failed")
