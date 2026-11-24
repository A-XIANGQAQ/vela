#!/bin/sh
# r7 host driver (Git Bash): r7_driver.sh <worker name> <graph> ...  -> one h-qnn container r7_<worker>, log work/<worker>.log
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"; WK=$1
MSYS_NO_PATHCONV=1 docker run --rm --cpus 4 -m 8g -v C:/src/ext/fsr4r/PUB/stage/npu/graphs:/src:ro -v C:/src/ext/fsr4r/AUDIT/r7/ebs:/r7 -v C:/src/ext/fsr4r/H/sdk/x/qairt/2.47.0.260601:/qairt:ro -v C:/src/ext/fsr4vk:/fsr4vk:ro -v C:/src/ext/fsr4r/T2:/t2:ro -v C:/src/ext/fsr4r/NPUPIPE/ctx/p14/v73:/p14:ro --name r7_$WK h-qnn bash /r7/tools/r7_build.sh "$@" > C:/src/ext/fsr4r/AUDIT/r7/ebs/work/$WK.log 2>&1
