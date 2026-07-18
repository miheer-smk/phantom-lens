#!/bin/bash
set -u; cd /home/iiitn/Downloads/phantom-lens-main; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "Major Revision Results/00_logs/wilddeepfake_extract.py" > "Major Revision Results/00_logs/wdf_extract.log" 2>&1
touch features/wilddeepfake_test_53d.csv.done
