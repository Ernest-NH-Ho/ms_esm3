#!/bin/bash
SUDOPASS=$(cat ~/.lazy.txt)
# CURRENT_PATH=$(pwd)
set -e

for pfam in PF01156 PF00069 PF00311 PF00920 PF01008 PF01182; do
    ./lazy_run.sh --file_name uniprokb_AND_reviewed_true_2025_05_04_filtered_1700_xDisorder_split_generated_${pfam} --file_type tsv --num_seeds 30 --num_cpu 24
done

echo $SUDOPASS | sudo -S shutdown -P +2