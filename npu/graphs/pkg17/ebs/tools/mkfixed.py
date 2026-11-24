import sys, numpy as np
n = int(sys.argv[1]); i = np.arange(n, dtype=np.uint64); ((i * 7 + (i >> 12)) % 59).astype(np.uint8).tofile(sys.argv[2])
