#!/bin/bash
set -e

for file in input_pdb/*.pdb; do
    file_base=$(basename "$file")
    file_base_txt="${file_base%.pdb}"
    # echo $file
    # echo $file_base
    # echo $file_base_txt
    if [ ! -e "input_dssp/${file_base_txt}.dssp" ]; then
        echo $file
        mkdssp -i $file -o input_dssp/$file_base_txt.dssp &
    fi
done
