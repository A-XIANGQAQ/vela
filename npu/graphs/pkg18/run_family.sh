#!/bin/sh
# usage: FAM=anf540 LW=960 LH=540 sh run_family.sh   (docker on the PC)
export PATH="/c/Program Files/Docker/Docker/resources/bin:$PATH"
MSYS_NO_PATHCONV=1 docker run --rm --cpus ${CPUS:-6} -m ${MEM:-6g} -v C:/src/ext/fsr4r/H:/h:ro -v C:/src/ext/fsr4r/ANF:/a \
  -e FAM=$FAM -e LW=$LW -e LH=$LH -e PAR=${PAR:-3} -e PHASES="$PHASES" -e GRID=${GRID:-8} --name impl_$FAM$GRID h-qnn bash /a/impl/graphs/build_family.sh
