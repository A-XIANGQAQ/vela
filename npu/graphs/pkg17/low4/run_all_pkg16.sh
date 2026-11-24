#!/bin/sh
# Package 16 low set: every shipped v73 graph planned for a 4 MB VTCM window, then the stored window size patched to 8 MB
# (VTCMLOW method: graphs/build7.sh recipe + patch_vtcm.py --mb 8; test 7 on node0112's Odin 2 Portal: F4p 3/3 correct).
# Git Bash on the PC:  sh run_all.sh > run_all.log 2>&1
# Steps: 1 ctx (17 graphs at vtcm_mb 4 + 2 controls at 8 MB that must equal the shipped files), 2 patch, 3 info + simulator
# on the 17 patched files. Two containers side by side (performance / ultraperf), own work folders.
# Inputs are read-only: H (SDK), NPUPIPE/b (the package-14 x86 model libraries), VTCMLOW (build7.sh, patch_vtcm.py).
export MSYS_NO_PATHCONV=1
L=C:/src/ext/fsr4r/AUDIT/r5/npu16/low; V=C:/src/ext/fsr4r/VTCMLOW
mkdir -p $L/graphs/ctx $L/patched $L/workA $L/workB
cp -p $V/graphs/build7.sh $L/build7.sh; cp -p $V/patch_vtcm.py $L/patch_vtcm.py
md5sum $L/build7.sh $L/patch_vtcm.py
run() { w=$1; shift; docker run --rm --cpus 6 -m 10g -v C:/src/ext/fsr4r/H:/h:ro -v $L/graphs:/g -v $L/$w:/w -v $L:/low:ro \
  -v C:/src/ext/fsr4r/RP6/test6/work/c6/libs:/t6l:ro -v C:/src/ext/fsr4r/NPUPIPE/b:/nb:ro "$@" h-qnn bash /low/build7.sh; }
PERF="nf4_exact nf4_k1h nf4_k1h_270 nf4_k1h_324 nf4_k1h_360 nf4_k1h_400 nf4_k1h_405 nf4_k1h_450 nf4_k1h_452"
UP="nf4up_k1h nf4up_k1h_270 nf4up_k1h_324 nf4up_k1h_360 nf4up_k1h_400 nf4up_k1h_405 nf4up_k1h_450 nf4up_k1h_452"
lib() { case $1 in nf4_exact) echo f:performance/nf4_exact/ht_t2s7p13;; nf4up_*) echo f:ultraperf/$1/k1h_a;; *) echo f:performance/$1/k1h_a;; esac; }
JA="nf4_k1h_360_v8,$(lib nf4_k1h_360),8"; for g in $PERF; do JA="$JA ${g}_v4,$(lib $g),4"; done
JB="nf4up_k1h_270_v8,$(lib nf4up_k1h_270),8"; for g in $UP; do JB="$JB ${g}_v4,$(lib $g),4"; done
echo "=== 1 ctx $(date +%T)"
run workA -e STEPS=ctx -e JOBS="$JA" --name low16_a > $L/workA/ctx.log 2>&1 &
run workB -e STEPS=ctx -e JOBS="$JB" --name low16_b > $L/workB/ctx.log 2>&1 &
wait
cat $L/workA/ctx.log $L/workB/ctx.log
echo "=== 1b controls: an 8 MB build here = the shipped file?"
for g in nf4_k1h_360 nf4up_k1h_270; do cmp -s $L/graphs/ctx/${g}_v8.bin C:/src/ext/fsr4r/NPUPIPE/ctx/p14/v73/$g.bin && echo "CONTROL $g v8 = shipped (md5 $(md5sum < $L/graphs/ctx/${g}_v8.bin | cut -c1-32))" || echo "CONTROL $g v8 DIFFERENT from the shipped file"; done
echo "=== 2 patch (stored window size 4 -> 8 MB) $(date +%T)"
for g in $PERF $UP; do python $L/patch_vtcm.py $L/graphs/ctx/${g}_v4.bin $L/patched/$g.bin --mb 8 2>&1 | tail -1; done
echo "=== 3 info + simulator on the patched files $(date +%T)"
SA=; for g in $PERF; do SA="$SA /low/patched/$g.bin"; done
SB=; for g in $UP; do SB="$SB /low/patched/$g.bin"; done
run workA -e STEPS="info sim" -e INFO="$SA" -e SIM="$SA" --name low16_a > $L/workA/sim.log 2>&1 &
run workB -e STEPS="info sim" -e INFO="$SB" -e SIM="$SB" --name low16_b > $L/workB/sim.log 2>&1 &
wait
cat $L/workA/sim.log $L/workB/sim.log
echo "=== 4 hashes against NPUPIPE/rp6test/ref.txt $(date +%T)"
for g in $PERF $UP; do want=$(grep "^$g " C:/src/ext/fsr4r/NPUPIPE/rp6test/ref.txt | cut -d' ' -f2); got=$(cat $L/workA/sim.log $L/workB/sim.log | grep "^SIM patched_$g\." | sed 's/.*check hash \([0-9a-f]*\).*/\1/' | tail -1)
  [ -n "$got" ] && [ "$got" = "$want" ] && echo "HASH OK   $g $got" || echo "HASH BAD  $g got '$got' want $want"; done
echo "=== sizes"; ls -l $L/patched | awk '{print $5, $9}'; md5sum $L/patched/*.bin
echo "=== done $(date +%T)"
