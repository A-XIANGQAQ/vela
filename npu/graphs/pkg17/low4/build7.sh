#!/bin/bash
# VTCMLOW (test 7) graphs, in the h-qnn image (QAIRT 2.47.0.260601 host tools; same recipe as RP6/test6/graphs/build6.sh).
# Mounts: /h = H (SDK, ro), /g = VTCMLOW/graphs (rw: ctx/ and patched/ live here), /w = VTCMLOW/work (rw),
#         /t6l = RP6/test6/work/c6/libs (ro: x86 libs of the copy graphs), /nb = NPUPIPE/b (ro: package-14 x86 model libraries).
# STEPS (env): ctx = context binaries for JOBS into /g/ctx; info = qnn-context-binary-utility JSON of INFO (paths of .bin);
#              sim = x86 HTP simulator on the checker's u8 input for SIM (paths of .bin).
# JOBS (env): space-separated OUT,LIB,VTCM_MB; LIB = c:<copy graph> or f:<init>/<G>/<lib name>. soc 43, v73, O 3, --log_level debug.
R=/h/sdk/x/qairt/2.47.0.260601
mkdir -p /tmp/cb; ln -sf /usr/bin/clang++-14 /tmp/cb/clang++; ln -sf /usr/bin/clang-14 /tmp/cb/clang
export QNN_SDK_ROOT=$R PYTHONPATH=$R/lib/python LD_LIBRARY_PATH=$R/lib/x86_64-linux-clang PATH=/tmp/cb:$R/bin/x86_64-linux-clang:$PATH
W=/w/c7; mkdir -p $W/cfg $W/log $W/sim $W/info /g/ctx; cd $W
ST=${STEPS:-ctx}
has() { case " $ST " in *" $1 "*) return 0;; esac; return 1; }
libpath() { case $1 in c:*) n=${1#c:}; echo "/t6l/x86_64-linux-clang/lib$n.so $n";; f:*) p=${1#f:}; echo "/nb/$(dirname $p)/libs/x86_64-linux-clang/lib$(basename $p).so $(basename $p)";; esac; }
if has ctx; then
  for j in $JOBS; do IFS=, read o lib vt <<< "$j"; set -- $(libpath $lib); so=$1; gn=$2
    echo "{\"graphs\": [{\"graph_names\": [\"$gn\"], \"vtcm_mb\": $vt, \"O\": 3}], \"devices\": [{\"soc_model\": 43, \"dsp_arch\": \"v73\"}]}" > cfg/htp_$o.json
    echo "{\"backend_extensions\": {\"shared_library_path\": \"$R/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$W/cfg/htp_$o.json\"}}" > cfg/ctx_$o.json
    t0=$(date +%s)
    qnn-context-binary-generator --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --model $so --binary_file $o --output_dir /g/ctx --config_file cfg/ctx_$o.json --log_level debug > log/ctx_$o.log 2>&1 \
      || { echo "CTX $o FAIL ($lib vtcm $vt): $(grep -iE 'error|not sufficiently|requires' log/ctx_$o.log | grep -v DEBUG | head -2 | cut -c1-220 | tr '\n' ' ')"; continue; }
    echo "CTX $o.bin ($lib vtcm_mb $vt, $(( $(date +%s) - t0 )) s) md5=$(md5sum < /g/ctx/$o.bin | cut -c1-32) size=$(stat -c %s /g/ctx/$o.bin) $(grep -E '^(spill|fill)_bytes' log/ctx_$o.log | tr '\n' ' ')"
  done
fi
if has info; then
  for b in $INFO; do n=$(basename $(dirname $b))_$(basename $b .bin)
    qnn-context-binary-utility --context_binary $b --json_file info/$n.json > info/$n.log 2>&1 || { echo "INFO $n FAIL: $(tail -2 info/$n.log | tr '\n' ' ')"; continue; }
    python3 - info/$n.json $n <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["info"]
for g in d["graphs"]:
    i = g["info"]; b = i.get("graphBlobInfo", {}).get("info", {}); b2 = i.get("graphBlobInfoV2", {})
    print("INFO %s graph %s vtcmSize=%s O=%s spillFill=%s ddrTensor=%s ioTensor=%s" % (sys.argv[2], i["graphName"], b.get("vtcmSize"),
          b.get("optimizationLevel"), b.get("spillFillBufferSize"), b2.get("ddrTensorSize"), b2.get("ioTensorSize")))
PY
  done
fi
if has sim; then
  [ -f /tmp/fnv ] || { printf '#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\nint main(int c,char**v){uint64_t hh=1469598103934665603ULL;for(int k=1;k<c;k++){FILE*f=fopen(v[k],"rb");if(!f)return 1;uint64_t h=1469598103934665603ULL;int ch;while((ch=getc_unlocked(f))!=EOF){h^=(uint8_t)ch;h*=1099511628211ULL;}fclose(f);hh^=h+(uint64_t)k;}printf("%%016llx\\n",(unsigned long long)hh);return 0;}\n' > /tmp/fnv.c && gcc -O2 -o /tmp/fnv /tmp/fnv.c; }
  echo '{"graphs": [{"graph_names": ["x"], "vtcm_mb": 8, "O": 3}], "devices": [{"soc_model": 43, "dsp_arch": "v73"}]}' > cfg/sim_htp.json
  echo "{\"backend_extensions\": {\"shared_library_path\": \"$R/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$W/cfg/sim_htp.json\"}}" > cfg/sim_ctx.json
  for b in $SIM; do o=$(basename $(dirname $b))_$(basename $b .bin)
    rows=$(qnn-context-binary-utility --context_binary $b --json_file sim/$o.info.json > /dev/null 2>&1; python3 -c "
import json; d=json.load(open('sim/$o.info.json'))['info']['graphs'][0]['info']['graphInputs'][0]['info']['dimensions']; print(d[1])")
    x=sim/x_$rows.raw; [ -f $x ] || python3 -c "
import numpy as np; n=$rows*480*32; i=np.arange(n,dtype=np.uint64); ((i*7+(i>>12))%59).astype(np.uint8).tofile('$x')"
    o=$o.$(date +%s%N); mkdir -p sim/$o; echo "x:=$W/$x" > sim/$o/list.txt; t0=$(date +%s)   # fresh folder per run, nothing deleted
    qnn-net-run --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --retrieve_context $b --input_list sim/$o/list.txt \
      --use_native_input_files --use_native_output_files --output_dir sim/$o/out --config_file cfg/sim_ctx.json --log_level warn > sim/$o/run.log 2>&1 || { echo "SIM $o FAIL"; tail -5 sim/$o/run.log; continue; }
    f=$(find sim/$o/out -name '*.raw' | sort | head -1); n=$(find sim/$o/out -name '*.raw' | wc -l)
    r=; if [ "$(stat -c %s "$f")" = "$(stat -c %s $x)" ] && cmp -s "$f" $x; then r="; SAME as the input (copy)"; fi
    echo "SIM $o ($(( $(date +%s) - t0 )) s): $n output(s), $(basename $f) $(stat -c %s "$f") bytes md5 $(md5sum < "$f" | cut -c1-32) check hash $(/tmp/fnv "$f")$r"
  done
fi
