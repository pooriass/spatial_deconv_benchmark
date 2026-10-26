#!/bin/bash
# Example driver. Replace TAGs, epochs and the depth numbers with your own.
# VIS_MEAN / VIS_SD = mean and SD of total UMI per spot in YOUR real Visium sections.
export DECONV_ROOT=$PWD/work
export NTHREADS=8

run () {   # run <TAG> <STATE_COL> <SUFFIX>
  TAG=$1 STATE_COL=$2 SUF=$3 SC_EPOCHS=300 ST_EPOCHS=250 \
  VIS_MEAN=8000 VIS_SD=6000 RESOLV_CAP=3000 BATCH=1024 \
  python resolvability.py
}

# one setting = one reference export (a gene space) x one label resolution
run A state ""                 # fine labels
run A state_coarse _coarse     # coarse labels
run B state ""
run B state_coarse _coarse

# score everything
RES=$DECONV_ROOT/results OUTDIR=$DECONV_ROOT/scores \
CONFIGS="A:,A:_coarse,B:,B:_coarse" PREFIXES="CAF_,DUCTAL_" CUT=0.20 \
python score.py
