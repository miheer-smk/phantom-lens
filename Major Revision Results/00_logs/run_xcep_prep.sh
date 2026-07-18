#!/bin/bash
set -u; cd /home/iiitn/Downloads/phantom-lens-main; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 "Major Revision Results/00_logs/xception_prep.py" --set ffpp --manifest data_xception/manifest_ffpp.csv --workers 10 > "Major Revision Results/00_logs/xcep_prep_ffpp.log" 2>&1
touch data_xception/ffpp_crops.done
python3 "Major Revision Results/00_logs/xception_prep.py" --set celebdf --manifest data_xception/manifest_celebdf.csv --workers 10 > "Major Revision Results/00_logs/xcep_prep_celebdf.log" 2>&1
touch data_xception/celebdf_crops.done
