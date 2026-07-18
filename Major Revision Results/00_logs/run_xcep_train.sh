#!/bin/bash
set -u; cd /home/iiitn/Downloads/phantom-lens-main; source .venv/bin/activate
until [ -f data_xception/ffpp_crops.done ] && [ -f data_xception/celebdf_crops.done ]; do sleep 60; done
python3 "Major Revision Results/00_logs/xception_train.py" > "Major Revision Results/00_logs/xcep_train.log" 2>&1
touch data_xception/xcep_train.done
bash "Major Revision Results/00_logs/backup_results.sh"
