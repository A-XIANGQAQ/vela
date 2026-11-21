#!/usr/bin/env python3
"""Compare regenerated parameter files (regen.sh OUT_DIR) with the committed ones in npu/graphs/params/.
.pkl: byte for byte, and if the bytes differ, the loaded objects (equal objects give the same graph).
.npz / .npy: every array, with its dtype and shape (an .npz is a zip with file times inside, so its bytes always differ).
Usage: check.py OUT_DIR [performance] [ultraperf]. Exit code 1 if any file differs or is missing."""
import sys, os, hashlib, pickle
import numpy as np

out = sys.argv[1]; presets = sys.argv[2:] or ['performance', 'ultraperf']
ref = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'params')
FILES = {'performance': ['params_n1.pkl', 'p13_w.npz', 'k1_w.npz'], 'ultraperf': ['params_n1.pkl', 'p13_w.npz', 'k1_w.npz', 'p07_sh.npy']}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
bad = 0
for P in presets:
    for f in FILES[P]:
        a, b = os.path.join(out, P, f), os.path.join(ref, P, f)
        if not os.path.exists(a):
            print('%-11s %-13s MISSING' % (P, f)); bad += 1; continue
        if f.endswith('.pkl'):
            if open(a, 'rb').read() == open(b, 'rb').read():
                r = 'byte-equal'
            elif pickle.load(open(a, 'rb')) == pickle.load(open(b, 'rb')):
                r = 'objects equal (bytes differ)'
            else:
                r = 'DIFFERS'
        else:
            A, B = np.load(a), np.load(b)
            if f.endswith('.npy'):
                A, B = {'': A}, {'': B}
            ka, kb = sorted(A.keys()) if hasattr(A, 'keys') else [], sorted(B.keys()) if hasattr(B, 'keys') else []
            diff = [k for k in sorted(set(ka) | set(kb)) if k not in ka or k not in kb or A[k].dtype != B[k].dtype
                    or A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])]
            r = ('%d of %d arrays equal' % (len(kb), len(kb))) if not diff else 'DIFFERS: ' + ' '.join(diff)
        ok = 'DIFF' not in r
        bad += not ok
        print('%-11s %-13s %-34s new %s  committed %s' % (P, f, r, sha(a)[:16], sha(b)[:16]))
print('RESULT', 'all equal' if not bad else '%d file(s) differ' % bad)
sys.exit(1 if bad else 0)
