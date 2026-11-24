"""VERJ1 (2026-11-22, fx38): assets/sgsr3/VERSION.json describes the OptiScaler DLL the plugin ships (md5, build date,
the fork's patch list, the live-model booleans). Checks: the file parses; the md5 is 32 lowercase hex characters; the
patch list has no duplicates, every entry matches ^\\d{4}[a-z]?$ and is in ascending order (string compare); it holds
0001-0059 (0009 and 0022 not in the build) with 0024b-d and 0043a-c instead of 0043; vk_live / dittosr_live / fsr3_live / anf_live (0.9.0) are booleans;
when OptiScaler.dll is present its md5 equals the one the file claims (the check is skipped when the binary is not
in this checkout).
Run: python3 -I tests/test_version_json.py"""
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VJ = os.path.join(ROOT, "assets", "sgsr3", "VERSION.json")

checks = 0


def ok(cond, what):
    global checks
    checks += 1
    assert cond, what


data = json.load(open(VJ, encoding="utf-8"))
ok(isinstance(data, dict), "VERSION.json parses as a JSON object")
ok(re.fullmatch(r"[0-9a-f]{32}", data["md5"]) is not None, data["md5"])
ok(re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["built"]) is not None, data["built"])

patches = data["patches"]
ok(isinstance(patches, list) and len(set(patches)) == len(patches), "patches: no duplicates")
ok(all(re.fullmatch(r"\d{4}[a-z]?", p) for p in patches), [p for p in patches if not re.fullmatch(r"\d{4}[a-z]?", p)])
ok(patches == sorted(patches), "patches in ascending order")
expected = {f"{n:04d}" for n in range(1, 60) if n not in (9, 22, 43)} | {"0024b", "0024c", "0024d", "0043a", "0043b", "0043c"}
ok(set(patches) == expected, sorted(set(patches) ^ expected))

ok(all(isinstance(data[k], bool) for k in ("vk_live", "dittosr_live", "fsr3_live", "anf_live")),
   {k: data.get(k) for k in ("vk_live", "dittosr_live", "fsr3_live", "anf_live")})

dll = os.path.join(ROOT, "assets", "sgsr3", "OptiScaler.dll")
if os.path.isfile(dll):
    ok(hashlib.md5(open(dll, "rb").read()).hexdigest() == data["md5"], "OptiScaler.dll md5 matches VERSION.json")
else:
    print("test_version_json: OptiScaler.dll not present, the dll md5 check is skipped")

print(f"test_version_json: {checks} checks, 0 failed")
