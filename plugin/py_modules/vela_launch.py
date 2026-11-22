#!/usr/bin/python3 -I
"""vela-launch <command> [args...]: starts a game with its Vela kit, without writing into the game folder.

The plugin keeps a kit per game in its settings folder (<kits>/<appid>/: files/ with OptiScaler and the FSR 4 provider,
launch.json with the game folder and Vela's options). This launcher shows the kit's files inside the game's exe folder
in a private mount namespace (bwrap, the same way as Armada's armada-mod-launch), sets Vela's environment and runs the
game. Without a kit for the game, or when anything about the kit or the mount is wrong, it runs the command as it is.

It is installed by the plugin to ~/.local/share/vela/bin/vela-launch and needs nothing from the plugin, so a game keeps
starting after the plugin is uninstalled. It runs from Steam's launch options (vela-launch ... %command%) or, on Armada
builds with plugin launch wrappers, from armada-game-launch (plugin.json "armada": {"launch": {"wrapper": ...}}).

CRASHREP (0.8.23): for a game whose kit is on, the game's stderr (Proton / Wine: err: lines, "Unhandled exception ... in
64-bit code", the crash backtrace with module names) is also kept in <kits>/<appid>/stderr-YYYYMMDD-HHMMSS.log (the run's
start, local time; the last STDERR_RUNS runs, each capped at STDERR_CAP), so a problem report can say where a game crashed.
The game is still the process Steam started (exec, as before): its exit code and signals are its own. Its stderr is a
pipe to a detached copier that writes every chunk on to the stderr Steam gave and to the file; nothing is added to the
game's environment that changes what it does (no WINEDEBUG, no PROTON_LOG: only what it prints anyway is kept)."""
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time

KITS = os.path.expanduser("~/homebrew/settings/Vela/kits")  # the plugin writes its own path here when it installs this file
BWRAP = "bwrap"
CMD = "%command%"
DONE = "VELA_LAUNCH_DONE"  # set for the game: a second pass (launch options and the Armada wrapper) adds nothing
APPID_KEYS = ("STEAM_COMPAT_APP_ID", "SteamAppId", "SteamGameId")
APPEND = {"WINEDLLOVERRIDES": ";"}  # variables Vela adds to instead of replacing
LOG_MAX = 256 * 1024
MAX_BINDS = 4000  # bwrap takes 9000 arguments at most, the game's command line included
# CRASHREP: the game's stderr per run (see the module text)
TEE = "VELA_LAUNCH_STDERR"  # set for the game once its stderr is kept: a second pass keeps nothing twice
STDERR_RUNS = 10            # runs kept per game; older stderr-*.log files of the game go (no other file is touched)
STDERR_CAP = 2 << 20        # one file at most: the first STDERR_HEAD bytes (Proton's start) + the newest output
STDERR_HEAD = 256 << 10
STDERR_CHUNK = 64 << 10
STDERR_RX = re.compile(r"^stderr-(\d{8})-(\d{6})\.log$")


def log(msg, appid=None):
    """One line in the kit's launch.log (Vela's settings folder) and on stderr (Steam's log)."""
    print(f"vela-launch: {msg}", file=sys.stderr, flush=True)
    try:
        d = os.path.join(KITS, appid) if appid else KITS
        if not os.path.isdir(d):
            return
        path = os.path.join(d, "launch.log")
        if os.path.isfile(path) and os.path.getsize(path) > LOG_MAX:
            os.replace(path, path + ".1")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def app_ids(env):
    """Every appid the environment names, in order. 0.8.22: a non-Steam shortcut may get "0" in SteamAppId and a 64-bit
    SteamGameId (appid << 32 | flags); its Proton prefix is still compatdata/<appid>, so that folder's name counts too."""
    out = []

    def add(value):
        value = str(value).strip()
        if value.isdigit() and value != "0" and value not in out:
            out.append(value)

    for key in APPID_KEYS:
        value = env.get(key, "").strip()
        if key == "SteamGameId" and value.isdigit() and int(value) >= 2 ** 32:
            value = str(int(value) >> 32)
        add(value)
    add(os.path.basename(env.get("STEAM_COMPAT_DATA_PATH", "").strip().rstrip("/")))
    return out


def app_id(env, kits=None):
    """The game's appid: the first one the environment names that has a kit, else the first one named."""
    ids = app_ids(env)
    for value in ids:
        if os.path.isfile(os.path.join(kits or KITS, value, "launch.json")):
            return value
    return ids[0] if ids else None


def is_assignment(word):
    name, sep, _ = word.partition("=")
    return bool(sep) and bool(name) and (name[0].isalpha() or name[0] == "_") and \
        all(c.isalnum() or c == "_" for c in name)


def read_kit(appid, kits=None):
    """The game's launch.json, or None when the game has no kit that is on."""
    path = os.path.join(kits or KITS, appid, "launch.json")
    try:
        with open(path, encoding="utf-8") as f:
            kit = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise RuntimeError(f"launch.json unreadable: {e}")
    if not isinstance(kit, dict) or not kit.get("on"):
        return None
    for key in ("game_dir", "layer", "work_dir"):
        if not isinstance(kit.get(key), str) or not os.path.isabs(kit[key]):
            raise RuntimeError(f"launch.json: bad {key}")
    return kit


def split_options(options):
    """Vela's options for the game (env before %command%, game arguments after it) -> (env dict, args)."""
    words = shlex.split(options or "")
    ci = words.index(CMD) if CMD in words else len(words)
    env = {}
    for w in words[:ci]:
        if is_assignment(w):
            name, _, value = w.partition("=")
            env[name] = value
    return env, [w for w in words[ci + 1:]]


def merge_env(env, ours):
    """The environment for the game: the player's own value of a variable wins, WINEDLLOVERRIDES gets Vela's entries
    after his (joined with ';')."""
    out = dict(env)
    for name, value in ours.items():
        sep = APPEND.get(name)
        have = out.get(name)
        if sep and have:
            if value not in have.split(sep):
                out[name] = have + sep + value
        elif not have:
            out[name] = value
    return out


def mounts(game, layer):
    """bwrap arguments that show the kit (layer) in the game folder. The game folder is the overlay's upper layer, so the
    game's own writes land in the install as before. Every kit entry is then mounted read-only over its name: otherwise a
    file of the game with the same name (an older amd_fidelityfx_dx12.dll) would win over the kit's, and a write to a kit
    file would leave a copy in the install. A folder the game does not have is mounted as a whole, read-only."""
    binds = []

    def walk(rel):
        for name in sorted(os.listdir(os.path.join(layer, rel))):
            sub = os.path.join(rel, name)
            top, real = os.path.join(layer, sub), os.path.join(game, sub)
            if os.path.islink(top):
                raise RuntimeError(f"{sub} is a link in the kit")
            if not os.path.isdir(top):
                if os.path.isdir(real):
                    raise RuntimeError(f"{sub} is a file in the kit and a folder in the game")
                binds.extend(["--ro-bind", top, real])
            elif os.path.isdir(real):
                walk(sub)
            elif os.path.lexists(real):
                raise RuntimeError(f"{sub} is a folder in the kit and a file in the game")
            else:
                binds.extend(["--ro-bind", top, real])

    walk("")
    if not binds:
        raise RuntimeError("the kit has no files")
    if len(binds) > MAX_BINDS:
        raise RuntimeError("the kit has more files than can be mounted one by one")
    return binds


def bwrap_argv(game, layer, work, bwrap=BWRAP):
    """bwrap ... -- (the game's command goes after it). The work folder must be on the game folder's filesystem and
    outside it: the plugin puts it next to the game's install folder."""
    return [bwrap, "--dev-bind", "/", "/", "--overlay-src", layer, "--overlay", game, work, game,
            *mounts(game, layer), "--"]


def clean_env(env=None):
    """V11 (audit 2026-11-09): a copy of the environment without the library and Python paths Decky's frozen Python exports
    to the plugin process: a system program (bwrap) started with them could load Decky's bundled libraries and fail to start."""
    return {k: v for k, v in (os.environ if env is None else env).items()
            if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH")}


def probe(argv, run=subprocess.run, timeout=30, env=None):
    """The same mounts with `true` as the program: (True, "") when they work, else (False, why).
    env (V11): the environment for bwrap; None = this process's (the launcher itself: Steam's)."""
    try:
        r = run([*argv, "true"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                timeout=timeout, **({} if env is None else {"env": env}))
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", "replace").strip() if isinstance(r.stderr, bytes) else str(r.stderr or "")
        return False, err.splitlines()[-1] if err else f"exit {r.returncode}"
    return True, ""


API_DX12 = ("-dx12", "-d3d12")
API_DX11 = ("-dx11", "-d3d11")


def api_flags(args):
    """The standalone API flags in the game's command (Steam puts a launch entry's arguments there), lower case."""
    return {a.lower() for a in args if a.lower() in API_DX12 + API_DX11}


def plan(args, env, kits=None, run=subprocess.run):
    """(command, environment) with the game's kit, or None: no kit for this game. Raises when the kit cannot be used."""
    appid = app_id(env, kits)
    if not appid:
        return None
    kit = read_kit(appid, kits)
    if kit is None:
        return None
    game, layer, work = kit["game_dir"], kit["layer"], kit["work_dir"]
    if not os.path.isdir(game):
        raise RuntimeError(f"game folder {game} not found")
    if not os.path.isdir(layer):
        raise RuntimeError("kit folder not found")
    missing = [rel for rel in kit.get("files") or () if not os.path.isfile(os.path.join(layer, *rel.split("/")))]
    if missing:
        raise RuntimeError(f"kit incomplete: {', '.join(missing)}")
    ours, extra = split_options(kit.get("options"))
    if kit.get("dx12_only"):  # DX12O: this kit runs only on DX12; never -dx12 on top of -dx11 (a fatal UE error)
        api = api_flags(args)
        if api & set(API_DX11):
            raise RuntimeError(f"the game starts with {sorted(api & set(API_DX11))[0]}: this game's kit runs only on DX12 "
                               "(the game runs without Vela's kit)")
        if not api and not api_flags(extra):
            extra = extra + ["-dx12"]
    os.makedirs(work, exist_ok=True)
    argv = bwrap_argv(game, layer, work, kit.get("bwrap") or BWRAP)
    ok, why = probe(argv, run)
    if not ok:
        raise RuntimeError(f"overlay probe failed: {why}")
    out = merge_env(env, ours)
    out[DONE] = "1"
    extra = [a for a in extra if a not in args]
    log(f"kit on: {game} ({(len(argv) - 11) // 3} entries), env {sorted(ours)}, args {extra}", appid)
    return [*argv, *args, *extra], out


# ---- CRASHREP: the game's stderr, kept per run ------------------------------------------------------------------------

def stderr_dir(env, kits=None):
    """The kit folder of a game whose kit is on (its stderr is kept there), else None. Never raises."""
    try:
        appid = app_id(env, kits)
        if not appid:
            return None
        d = os.path.join(kits or KITS, appid)
        with open(os.path.join(d, "launch.json"), encoding="utf-8") as f:
            kit = json.load(f)
        return d if isinstance(kit, dict) and kit.get("on") else None
    except Exception:
        return None


def stderr_path(d, t=None):
    """<kit>/stderr-YYYYMMDD-HHMMSS.log: the run's start, local time (the kept OptiScaler runs are named the same way)."""
    return os.path.join(d, time.strftime("stderr-%Y%m%d-%H%M%S.log", time.localtime(t)))


def rotate_stderr(d, keep=STDERR_RUNS - 1):
    """Only the newest `keep` stderr-*.log files of the folder stay (the new run's file makes STDERR_RUNS); nothing else goes."""
    try:
        names = sorted(n for n in os.listdir(d) if STDERR_RX.match(n))
    except OSError:
        return
    for n in names[:max(0, len(names) - keep)]:
        try:
            os.remove(os.path.join(d, n))
        except OSError:
            pass


def _trim(f, cap=STDERR_CAP, head=STDERR_HEAD):
    """A file over `cap`: its first `head` bytes (to a line end), a cut line, and the newest cap/2 bytes (from a line start)."""
    size = f.seek(0, 2)
    if size <= cap:
        return
    f.seek(0)
    h = f.read(head)
    nl = h.rfind(b"\n")
    hend = nl + 1 if nl > 0 else len(h)
    keep = cap // 2
    f.seek(size - keep)
    tail = f.read(keep)
    nl = tail.find(b"\n")
    if 0 <= nl < 4096:
        tail = tail[nl + 1:]
    f.seek(hend)
    f.write(f"... [vela-launch: output cut here; the file keeps the start and the newest {keep >> 10} KB] ...\n".encode() + tail)
    f.truncate()


def _write_all(fd, data):
    view = memoryview(data)
    stalls = 0
    while view:
        try:
            n = os.write(fd, view)
        except BlockingIOError:  # a non-blocking stderr that is full: wait a little, never forever
            stalls += 1
            if stalls > 200:
                raise
            time.sleep(0.005)
            continue
        view = view[n:]


def pump(rfd, out_fd, path, cap=STDERR_CAP, head=STDERR_HEAD):
    """Copies everything from rfd to out_fd (first: the stderr Steam gave sees it as before) and into `path` (capped), until
    the last writer closes the pipe. A failing out_fd or file is dropped; the pipe is drained to its end either way, so the
    game never blocks or gets SIGPIPE from a dead reader."""
    f = None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0), 0o644)
        f = os.fdopen(fd, "r+b", buffering=0)
        f.seek(0, 2)
    except Exception:
        f = None
    out_ok = out_fd is not None
    while True:
        try:
            data = os.read(rfd, STDERR_CHUNK)
        except InterruptedError:
            continue
        except OSError:
            break
        if not data:
            break
        if out_ok:
            try:
                _write_all(out_fd, data)
            except Exception:
                out_ok = False
        if f is not None:
            try:
                f.write(data)
                if f.tell() > cap:
                    _trim(f, cap, head)
            except Exception:
                try:
                    f.close()
                except Exception:
                    pass
                f = None
    if f is not None:
        try:
            f.close()
        except Exception:
            pass


def _tee_child(r, path):
    """The copier (a grandchild of the launcher, so the game never sees a child it did not start): its own session, deaf to
    the stop signals the game gets (it ends when the game's last process closes stderr), only the pipe and the original
    stderr open. Never returns."""
    try:
        os.setsid()
        for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT", "SIGPIPE", "SIGUSR1", "SIGUSR2"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_IGN)
        dn = os.open(os.devnull, os.O_RDWR)
        os.dup2(dn, 0)
        os.dup2(dn, 1)
        try:
            top = min(os.sysconf("SC_OPEN_MAX"), 1 << 16)
        except (OSError, ValueError):
            top = 4096
        os.closerange(3, r)
        os.closerange(r + 1, max(top, r + 2))
        rotate_stderr(os.path.dirname(path))
        pump(r, 2, path)
    except BaseException:
        pass
    finally:
        os._exit(0)


def start_tee(path):
    """fd 2 of this process (and so of the game it execs) becomes a pipe to the copier; the stderr Steam gave keeps getting
    every byte. True when on; False (fd 2 untouched) when anything fails or the platform has no fork (the tests' Windows)."""
    if not path or not hasattr(os, "fork"):
        return False
    try:
        import fcntl
        os.fstat(2)  # no stderr at all: nothing to keep
        r, w = os.pipe()
        for old in (r, w):  # never on 0 / 1 / 2 (a launch with stdin closed would get the pipe there)
            if old < 3:
                new = fcntl.fcntl(old, fcntl.F_DUPFD_CLOEXEC, 10)
                os.close(old)
                r, w = (new, w) if old == r else (r, new)
        try:
            fcntl.fcntl(w, getattr(fcntl, "F_SETPIPE_SZ", 1031), 1 << 20)  # room for a burst while the file is written
        except OSError:
            pass
    except Exception:
        return False
    # an ignored SIGCHLD (inherited from the caller) would reap the helper before waitpid sees it: default while it runs,
    # then back as it was, so the game inherits the same disposition as before
    chld = getattr(signal, "SIGCHLD", None)
    old_chld = signal.getsignal(chld) if chld else None
    if chld and old_chld == signal.SIG_IGN:
        signal.signal(chld, signal.SIG_DFL)
    try:
        try:
            pid = os.fork()
        except OSError:
            os.close(r)
            os.close(w)
            return False
        if pid == 0:
            code = 1
            try:
                os.close(w)
                if os.fork() == 0:
                    _tee_child(r, path)
                code = 0
            except BaseException:
                pass
            finally:
                os._exit(code)
        os.close(r)
        try:
            _, status = os.waitpid(pid, 0)
            alive = os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        except OSError:
            alive = False
    finally:
        if chld and old_chld == signal.SIG_IGN:
            signal.signal(chld, signal.SIG_IGN)
    try:
        if alive:
            sys.stderr.flush()
            os.dup2(w, 2)
    except OSError:
        alive = False
    finally:
        os.close(w)
    return alive


def keep_stderr(env=None, kits=None):
    """CRASHREP: the game's stderr into its kit folder for this run (POSIX; once per launch; never in the way of the game)."""
    env = os.environ if env is None else env
    if os.name != "posix" or env.get(DONE) or env.get(TEE):
        return False
    d = stderr_dir(env, kits)
    if d and start_tee(stderr_path(d)):
        env[TEE] = "1"
        return True
    return False


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.exit("usage: vela-launch <command> [args...]")
    # NAME=value words in front of the command: the player's own variables typed before %command% in Steam's options
    while args and is_assignment(args[0]):
        name, _, value = args.pop(0).partition("=")
        os.environ[name] = value
    if not args:
        sys.exit("vela-launch: no command")
    try:
        keep_stderr()
    except Exception:
        pass
    planned = None
    if not os.environ.get(DONE):
        try:
            planned = plan(args, dict(os.environ))
        except Exception as e:  # never in the way of the game: it starts without Vela
            log(f"Vela left out of this launch: {e}", app_id(os.environ))
    if planned:
        cmd, env = planned
        try:
            os.execvpe(cmd[0], cmd, env)
        except (OSError, ValueError) as e:
            log(f"overlay launch failed: {e}", app_id(os.environ))
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
