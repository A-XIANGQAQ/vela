#!/bin/bash
# ANF graph family: one v73 context binary holding graphs <FAM>_p<k>, k in PHASES (default 0..63).
# Run inside the h-qnn image (QAIRT 2.47.0.260601). Mounts: /h (SDK, ro), /a (ANF, rw).
# env: FAM=anf540 LW=960 LH=540 [PHASES="0 1 .."] [PAR=6] [CIN=32] [VT=8]
R=/h/sdk/x/qairt/2.47.0.260601
mkdir -p /tmp/cb; ln -sf /usr/bin/clang++-14 /tmp/cb/clang++; ln -sf /usr/bin/clang-14 /tmp/cb/clang
export QNN_SDK_ROOT=$R PYTHONPATH=$R/lib/python LD_LIBRARY_PATH=$R/lib/x86_64-linux-clang PATH=/tmp/cb:$R/bin/x86_64-linux-clang:$PATH
MODEL=/a/data/assets/modelzoo/aiff-sr-v2s-prod-7p1-invpow2
W=/a/impl/work/${FAM}_g${GRID:-8}; mkdir -p $W /a/impl/graphs/ctx
PHASES=${PHASES:-$(seq -s' ' 0 63)}; CIN=${CIN:-32}
one() { j=$1; p=$j; [ "${GRID:-8}" = 4 ] && p=$(( (j/4*2+1)*8 + (j%4)*2+1 )); [ "${GRID:-8}" = 32 ] && p=$(( (j/8*2+1)*8 + j%8 )); m=${FAM}_p$j; d=$W/$m; mkdir -p $d/libs; cd $d
  [ -f libs/x86_64-linux-clang/lib$m.so ] && return 0
  python3 /a/src/gen_anf_onnx.py $d $m $LW $LH $p $CIN 0 $MODEL > gen.log 2>&1 || { echo "$m GEN FAIL"; return 1; }
  python3 $R/bin/x86_64-linux-clang/qnn-onnx-converter -i $m.onnx --custom_io $m.io.yaml --float_fallback --float_bitwidth 16 -o $m.cpp > convert.log 2>&1 || { echo "$m CONVERT FAIL"; return 1; }
  qnn-model-lib-generator -c $m.cpp -b $m.bin -l $m -t x86_64-linux-clang -o libs > libgen.log 2>&1 || { echo "$m LIBGEN FAIL"; return 1; }
  echo "ONNX $m md5=$(md5sum < $m.onnx | cut -c1-32) lib=$(md5sum < libs/x86_64-linux-clang/lib$m.so | cut -c1-32)"; }
export -f one; export FAM LW LH CIN MODEL W R
t0=$(date +%s)
echo $PHASES | tr ' ' '\n' | xargs -P ${PAR:-6} -I{} bash -c 'one {}'
echo "LIBS done $(( $(date +%s) - t0 )) s"
cd $W
L=; N=; for p in $PHASES; do L="$L,$W/${FAM}_p$p/libs/x86_64-linux-clang/lib${FAM}_p$p.so"; N="$N,\"${FAM}_p$p\""; done
L=${L#,}; N=${N#,}
echo "{\"graphs\": [{\"graph_names\": [$N], \"vtcm_mb\": ${VT:-8}, \"O\": 3}], \"devices\": [{\"soc_model\": 43, \"dsp_arch\": \"v73\"}]}" > htp.json
echo "{\"backend_extensions\": {\"shared_library_path\": \"$R/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$W/htp.json\"}}" > ctx.json
t0=$(date +%s)
qnn-context-binary-generator --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --model $L --binary_file ${FAM}s${GRID:-8} --output_dir /a/impl/graphs/ctx --config_file ctx.json --log_level warn > ctx.log 2>&1 \
  || { echo "CTX FAIL"; tail -5 ctx.log; exit 1; }
echo "CTX $FAM.bin ($(( $(date +%s) - t0 )) s) size=$(stat -c %s /a/impl/graphs/ctx/${FAM}s${GRID:-8}.bin) md5=$(md5sum < /a/impl/graphs/ctx/${FAM}s${GRID:-8}.bin | cut -c1-32)"
qnn-context-binary-utility --context_binary /a/impl/graphs/ctx/${FAM}s${GRID:-8}.bin --json_file info.json > info.log 2>&1 && python3 -c "
import json; d=json.load(open('info.json'))['info']; g=d['graphs']; print('graphs in context:', len(g), g[0]['info']['graphName'], '...', g[-1]['info']['graphName'], [t['info']['dimensions'] for t in g[0]['info']['graphInputs']], [t['info']['dimensions'] for t in g[0]['info']['graphOutputs']])"
