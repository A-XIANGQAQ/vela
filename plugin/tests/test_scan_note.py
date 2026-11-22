"""R822 (BG3 report AKG5XB: scan exe null, inputs [], no reason in the report): a scan that finds no game exe says why:
whether the folder is there, the walk's errors, how many entries it saw, and the names at the top and in bin/.
Run: python -I tests/test_scan_note.py"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "py_modules"))
import sgsr3  # noqa: E402

tmp = tempfile.mkdtemp(prefix="vela_scan_note_")
# a folder with no Windows exe (a game the walk cannot place)
g = os.path.join(tmp, "Baldurs Gate 3"); os.makedirs(os.path.join(g, "bin")); os.makedirs(os.path.join(g, "Data"))
open(os.path.join(g, "bin", "bg3.so"), "w").close(); open(os.path.join(g, "Data", "Gustav.pak"), "w").close()
sc = sgsr3.scan_dir(g)
assert sc["exe"] is None and "scan_note" in sc, sc
n = sc["scan_note"]
assert n["root_is_dir"] is True and n["errors"] == [] and n["entries"] >= 4, n
assert n["top"] == ["Data/", "bin/"] and n["bin"] == ["bg3.so"], n
# the folder Steam names is not there (SD card not mounted, renamed folder)
gone = sgsr3.scan_dir(os.path.join(tmp, "missing"))
assert gone["exe"] is None and gone["scan_note"]["root_is_dir"] is False and gone["scan_note"]["top"] == [], gone
# a found exe: no note (the scan stays as it was)
with open(os.path.join(g, "bin", "bg3.exe"), "wb") as f:  # pick_exe takes exes of 1 MB and more
    f.write(b"MZ"); f.truncate(2 * 1024 * 1024)
ok = sgsr3.scan_dir(g)
assert ok["exe"] == "bg3.exe" and "scan_note" not in ok, ok
# at most 60 names
many = os.path.join(tmp, "many"); os.makedirs(many)
for i in range(80):
    open(os.path.join(many, "f%02d.dat" % i), "w").close()
assert len(sgsr3.scan_dir(many)["scan_note"]["top"]) == 60
print("test_scan_note: ok")
