#!/bin/bash
# r7 E-BS for the shipped v73 graphs, in-container (h-qnn). One preset per container (the generator layout is global).
# Mounts: /src = PUB/stage/npu/graphs (ro), /r7 = AUDIT/r7/ebs, /qairt (ro), /fsr4vk (ro), /t2 = T2 (ro, inputs), /p14 = NPUPIPE/ctx/p14/v73 (ro).
# Usage: r7_build.sh <worker name> <graph> [<graph> ...]     z per layer: /r7/zset_<graph>.txt if present, else the preset default below.
set -u
WK=$1; shift; R=/qairt
mkdir -p /tmp/cb; ln -sf /usr/bin/clang++-14 /tmp/cb/clang++; ln -sf /usr/bin/clang-14 /tmp/cb/clang
export QNN_SDK_ROOT=$R PYTHONPATH=$R/lib/python LD_LIBRARY_PATH=$R/lib/x86_64-linux-clang PATH=/tmp/cb:$R/bin/x86_64-linux-clang:$PATH
gcc -O2 -o /tmp/fnv /r7/tools/fnv64.c 2>/dev/null || clang-14 -O2 -o /tmp/fnv /r7/tools/fnv64.c || { echo "FNV BUILD FAIL"; exit 1; }
ZDEF_performance="p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64"
ZDEF_ultraperf="p03e=160 p05e=160 p09e=160 p11e=160 p15e=160 p17e=160 p19e=160 p21e=160 p23e=160 p25e=160 p27e=160"
build() {  # build <dir> <flags> "<z set>"  -> <dir>/ctx/$G.bin, <dir>/libs/x86_64-linux-clang/lib$m.so
  local W=$1; mkdir -p $W/libs $W/ctx; cd $W
  ( for kv in $3; do t=${kv%%=*}; export ZPO_${t}D_o=${kv##*=}; done
    KIND=$KIND VARIANTS="$m:$2:0" INIT_BIN=/b/initializers.bin NSZ_W=${NSZ_W:-} NSZ_HN=${NSZ_HN:-} NSZ_HR=${NSZ_HR:-} NSZ_WR=${NSZ_WR:-} python3 $GEN $W > gen.log 2>&1 )
  [ -f $m.onnx ] || { echo "$G GEN FAIL $W"; tail -6 gen.log; return 1; }
  python3 $R/bin/x86_64-linux-clang/qnn-onnx-converter -i $m.onnx --custom_io $m.io.yaml --float_fallback --float_bitwidth 16 -o $m.cpp > convert.log 2>&1 || { echo "$G CONVERT FAIL $W"; tail -6 convert.log; return 1; }
  qnn-model-lib-generator -c $m.cpp -b $m.bin -l $m -t x86_64-linux-clang -o libs > libgen.log 2>&1 || { echo "$G LIBGEN FAIL $W"; tail -6 libgen.log; return 1; }
  echo "{\"graphs\": [{\"graph_names\": [\"$m\"], \"vtcm_mb\": 8, \"O\": 3}], \"devices\": [{\"soc_model\": 43, \"dsp_arch\": \"v73\"}]}" > htp_v73.json
  echo "{\"backend_extensions\": {\"shared_library_path\": \"$R/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so\", \"config_file_path\": \"$W/htp_v73.json\"}}" > ctxcfg_v73.json
  qnn-context-binary-generator --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --model libs/x86_64-linux-clang/lib$m.so --binary_file $G --output_dir $W/ctx \
    --config_file ctxcfg_v73.json > ctx_v73.log 2>&1 || { echo "$G CTX FAIL $W"; grep -iE error ctx_v73.log | head -3; return 1; }
}
sim() {  # sim <ctx> <input> <dir> -> prints the path of the output raw
  mkdir -p $3; echo "x:=$2" > $3/list.txt
  qnn-net-run --backend $R/lib/x86_64-linux-clang/libQnnHtp.so --retrieve_context $1 --input_list $3/list.txt --use_native_input_files \
    --use_native_output_files --output_dir $3/out --config_file $CFG --log_level warn > $3/run.log 2>&1
  find $3/out -name "*.raw" | sort | head -1
}
for G in "$@"; do t0=$(date +%s)
  row=$(awk -v g="$G" '$1 == g' /src/graphs.txt); [ -n "$row" ] || { echo "$G: not in graphs.txt"; continue; }
  read -r _ KIND PRESET m NSZ_W NSZ_HN NSZ_HR NSZ_WR <<< "$row"
  FLG=$(cat /src/flags/$PRESET.txt); EL="p03e p05e p09e p11e p15e p17e p19e p21e p23e p25e p27e"; rows=540
  if [ $KIND = ht ]; then FLG=$(cat /src/flags/exact.txt); EL="p03e p05e p09e p11e p15e p17e p19e p21e p23e p25e"; fi
  if [ $KIND = n7 ]; then rows=360; fi
  if [ $KIND = n9 ]; then rows=$NSZ_HR; fi
  S=/src/gen; P=/src/params/$PRESET
  mkdir -p /vt /ht /ht3 /n/n1 /gn /nf /k1h/build /nsz/tools /b
  cp /r7/tools/gen_r6.py /vt/gen_bp.py; cp $S/gen_nsz.py /nsz/tools/; cp $S/gen_ht3.py /ht3/; cp $S/gen_ht.py /ht/; cp $S/gen_n1.py /n/n1/
  cp $S/gen_n7.py $S/gen_n9.py /gn/; cp $S/gen_k1h.py /k1h/; cp $P/k1_w.npz /k1h/build/
  cp $S/gen_nf2.py $P/params_n1.pkl $P/p13_w.npz /nf/; cp /r7/tools/gen_nf_r6_$PRESET.py /nf/gen_nf.py
  if [ -f $P/p07_sh.npy ]; then cp $P/p07_sh.npy /nf/; fi
  cp /fsr4vk/assets/fsr411/general/1080/$PRESET/initializers.bin /b/initializers.bin
  GEN=/vt/gen_bp.py; if [ $KIND = n9 ]; then GEN=/nsz/tools/gen_nsz.py; fi
  if [ $PRESET = ultraperf ]; then ZALL=$ZDEF_ultraperf; else ZALL=$ZDEF_performance; fi
  if [ -f /r7/zset_$G.txt ]; then ZALL=$(cat /r7/zset_$G.txt); fi
  ZSET=""; EFL="$FLG"
  for kv in $ZALL; do t=${kv%%=*}; case " $EL " in *" $t "*) ZSET="$ZSET $kv"; EFL="$EFL+Z$t";; esac; done
  ZSET=${ZSET# }
  # 1. control: with the patch off the chain must give the released file
  C=/r7/work/ctl/$G; build $C "$FLG" "" || { echo "RESULT|$G|$PRESET|$KIND|CONTROL BUILD FAILED"; continue; }
  cmd5=$(md5sum < $C/ctx/$G.bin | cut -c1-32); rmd5=$(md5sum < /p14/$G.bin | cut -c1-32); cok=NO; if [ "$cmd5" = "$rmd5" ]; then cok=yes; fi
  cddr=$(grep -E "^(spill|fill)_bytes" $C/ctx_v73.log | tr "\n" " ")
  # 2. E-BS
  E=/r7/work/ebs/$G; build $E "$EFL" "$ZSET" || { echo "RESULT|$G|$PRESET|$KIND|EBS BUILD FAILED|ctl_ok=$cok"; continue; }
  emd5=$(md5sum < $E/ctx/$G.bin | cut -c1-32); esz=$(stat -c %s $E/ctx/$G.bin); ddr=$(grep -E "^(spill|fill|write_total|read_total)_bytes" $E/ctx_v73.log | tr "\n" " ")
  # 3. simulator: checker input (check hash vs ref.txt), noise and photo (vs the simulator output of the released file)
  CFG=$E/ctxcfg_v73.json; IN=/r7/work/in_$WK/$rows; mkdir -p $IN; n=$((rows*480*32))
  if [ ! -f $IN/fixed.raw ]; then python3 /r7/tools/mkfixed.py $n $IN/fixed.raw; fi
  for k in noise photo; do if [ ! -f $IN/$k.raw ]; then head -c $n /t2/work/inputs/$k.raw > $IN/$k.raw; fi; done
  f=$(sim $E/ctx/$G.bin $IN/fixed.raw $E/sim/fixed); hash=none; if [ -n "$f" ]; then hash=$(/tmp/fnv "$f"); fi
  ref=$(awk -v g="$G" '$1 == g {print $2}' /r7/ref.txt); hok=NO; if [ "$hash" = "$ref" ]; then hok=yes; fi
  oth=""; allok=yes
  for k in noise photo; do
    a=$(sim /p14/$G.bin $IN/$k.raw $E/sim/rel_$k); b=$(sim $E/ctx/$G.bin $IN/$k.raw $E/sim/$k)
    if [ -n "$a" ] && [ -n "$b" ] && cmp -s "$a" "$b"; then oth="$oth $k=0"
    else d=norun; if [ -n "$a" ] && [ -n "$b" ]; then d=$(cmp -l "$a" "$b" | wc -l); fi; oth="$oth $k=DIFF:$d"; allok=NO; fi
  done
  v=FAIL
  if [ $cok = yes ] && [ $hok = yes ] && [ $allok = yes ]; then v=PASS
    mkdir -p /r7/b/$PRESET/$G/libs/x86_64-linux-clang /r7/ctx8
    cp $E/libs/x86_64-linux-clang/lib$m.so /r7/b/$PRESET/$G/libs/x86_64-linux-clang/lib$m.so; cp $E/ctx/$G.bin /r7/ctx8/$G.bin
  fi
  echo "RESULT|$G|$PRESET|$KIND|$m|$rows|ctl=$cmd5|released=$rmd5|ctl_ok=$cok|ebs=$emd5|size=$esz|$ddr|ctl_ddr: $cddr|hash=$hash|ref=$ref|hash_ok=$hok|other:$oth|$v|z: $ZSET|s=$(( $(date +%s) - t0 ))"
done
echo "WORKER $WK DONE"
