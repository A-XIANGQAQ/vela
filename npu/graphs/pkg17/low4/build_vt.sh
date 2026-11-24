#!/bin/sh
# Round 7: the same sweep on the E-BS graphs (model libraries AUDIT/r7/ebs/b). Every shipped v73 graph is built for vtcm_mb 2, 3, 5, 6, 7 (4 exists
# in AUDIT/r5/npu16/low, 8 is the released file), then the stored window size is patched to 8 MB (VTCMLOW method), so every
# file runs with the same 8 MB window and only the compiler's plan differs.  Git Bash:  sh build_vt.sh > build_vt.log 2>&1
export MSYS_NO_PATHCONV=1
L=C:/src/ext/fsr4r/AUDIT/r7/vt; V=C:/src/ext/fsr4r/VTCMLOW
mkdir -p $L/graphs/ctx $L/patched $L/workA $L/workB
cp -p $V/graphs/build7.sh $L/build7.sh; cp -p $V/patch_vtcm.py $L/patch_vtcm.py
run() { w=$1; shift; docker run --rm --cpus 6 -m 10g -v C:/src/ext/fsr4r/H:/h:ro -v $L/graphs:/g -v $L/$w:/w -v $L:/low:ro \
  -v C:/src/ext/fsr4r/RP6/test6/work/c6/libs:/t6l:ro -v C:/src/ext/fsr4r/AUDIT/r7/ebs/b:/nb:ro "$@" h-qnn bash /low/build7.sh; }
PERF="nf4_exact nf4_k1h nf4_k1h_270 nf4_k1h_324 nf4_k1h_360 nf4_k1h_400 nf4_k1h_405 nf4_k1h_450 nf4_k1h_452"
UP="nf4up_k1h nf4up_k1h_270 nf4up_k1h_324 nf4up_k1h_360 nf4up_k1h_400 nf4up_k1h_405 nf4up_k1h_450 nf4up_k1h_452"
SZ="${SZ:-2 3 5 6 7}"
lib() { case $1 in nf4_exact) echo f:performance/nf4_exact/ht_t2s7p13;; nf4up_*) echo f:ultraperf/$1/k1h_a;; *) echo f:performance/$1/k1h_a;; esac; }
JA=; for g in $PERF; do for v in $SZ; do JA="$JA ${g}_v$v,$(lib $g),$v"; done; done
JB=; for g in $UP; do for v in $SZ; do JB="$JB ${g}_v$v,$(lib $g),$v"; done; done
echo "=== ctx $(date +%T)"
run workA -e STEPS=ctx -e JOBS="$JA" --name vt7_a > $L/workA/ctx.log 2>&1 &
run workB -e STEPS=ctx -e JOBS="$JB" --name vt7_b > $L/workB/ctx.log 2>&1 &
wait
cat $L/workA/ctx.log $L/workB/ctx.log
echo "=== patch $(date +%T)"
for g in $PERF $UP; do for v in $SZ; do [ -f $L/graphs/ctx/${g}_v$v.bin ] && python $L/patch_vtcm.py $L/graphs/ctx/${g}_v$v.bin $L/patched/${g}_v$v.bin --mb 8 2>&1 | tail -1 | cut -c1-140; done; done
echo "=== done $(date +%T): $(ls $L/patched | wc -l) patched files"
