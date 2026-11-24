#!/bin/bash
# r6 micro-probe, in-container (h-qnn). Mounts: /r6 = AUDIT/r6/study, /ship = folder with the shipped k1h_a.onnx (ro), /qairt (ro).
# Usage: mp_in.sh <run name>   env: ZS, LAYERS, HW (see mp_e.py)
set -u
R=/qairt; N=$1; W=/r6/probe/$N
mkdir -p /tmp/cb $W; ln -sf /usr/bin/clang++-14 /tmp/cb/clang++; ln -sf /usr/bin/clang-14 /tmp/cb/clang
export QNN_SDK_ROOT=$R PYTHONPATH=$R/lib/python LD_LIBRARY_PATH=$R/lib/x86_64-linux-clang PATH=/tmp/cb:$R/bin/x86_64-linux-clang:$PATH
cd $W; t0=$(date +%s)
python3 /r6/tools/mp_e.py gen /ship/k1h_a.onnx $W > gen.log 2>&1 || { echo "MP GEN FAIL"; tail -5 gen.log; exit 1; }
python3 $R/bin/x86_64-linux-clang/qnn-onnx-converter -i mp.onnx --custom_io mp.io.yaml --float_fallback --float_bitwidth 16 -o mp.cpp > convert.log 2>&1 \
  || { echo "MP CONVERT FAIL"; tail -8 convert.log; exit 1; }
qnn-model-lib-generator -c mp.cpp -b mp.bin -l mp -t x86_64-linux-clang -o libs > libgen.log 2>&1 || { echo "MP LIBGEN FAIL"; tail -8 libgen.log; exit 1; }
echo '{"graphs": [{"graph_names": ["mp"], "vtcm_mb": 8, "O": 3}], "devices": [{"soc_model": 43, "dsp_arch": "v73"}]}' > htp.json
echo "{\"backend_extensions\": {\"shared_library_path\": \"$R/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$W/htp.json\"}}" > ctxcfg.json
qnn-context-binary-generator --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --model libs/x86_64-linux-clang/libmp.so --binary_file mp --output_dir $W/ctx \
  --config_file ctxcfg.json > ctx.log 2>&1 || { echo "MP CTX FAIL"; grep -iE "error" ctx.log | head -5; exit 1; }
echo "$W/x.raw" > list.txt
qnn-net-run --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --retrieve_context $W/ctx/mp.bin --input_list list.txt --use_native_input_files \
  --use_native_output_files --output_dir $W/sim --config_file ctxcfg.json --log_level warn > run.log 2>&1; echo "MP SIM rc=$? s=$(( $(date +%s) - t0 ))"
cat gen.log | grep -v wrote
python3 /r6/tools/mp_e.py cmp $W $W/sim 2>&1 | tee cmp.txt
