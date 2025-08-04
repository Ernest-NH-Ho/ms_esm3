#!/bin/bash
set -e

# Function to display help
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --file_name : Name of the uniprot dataframe"
  echo "  --file_type : Either tsv or csv (default)"
  echo "  --num_seeds : Do not input 1 or less"
  echo "  --num_cpu"
  echo "  --help"
  exit 0
}

# for arg in "$@"; do
#   if [[ "$arg" == "--help" ]]; then
#     show_help
#   fi
# done

# if no args at all, bail out
if [ $# -eq 0 ]; then
  echo "No arguments provided. Use --help for usage information."
  exit 1
fi

file_type="csv"
folder="scripts"
num_seeds=3
num_cpu=4

# parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      show_help
      ;;
    --file_name)
      file_name="$2"
      shift 2
      ;;
    --file_type)
      file_type="$2"
      shift 2
      ;;
    --num_seeds)
      num_seeds="$2"
      shift 2
      ;;
    --num_cpu)
      num_cpu="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

###########################################################
# Create dssp file
echo "1. Run dssp"
./$folder/1_dssp.sh

# # Create prompt for esm3
# echo "2. Prepare prompts from dssp outputs"
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $folder/2_processData.py -file_name $file_name -file_type $file_type

for ((rs = 41; rs <41+num_seeds; rs++)); do
    if [ ! -e "esm3_embeddings/rs${rs}" ]; then
        mkdir "esm3_embeddings/rs${rs}" &
    fi
done

for name in wt; do
    if [ ! -e "esm3_embeddings/${name}" ]; then
        mkdir "esm3_embeddings/${name}" &
    fi
done

## Generate mutants using esm3
echo "3. Starting running esm3"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $folder/3_runESM3_emb.py -file_name $file_name -num_seeds $num_seeds

## Calculate the difference in embedding and epistasis
echo "4. Calculating dataset from embeddings"
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python 4_calcDataset.py -file_name $file_name -num_seeds 5 -num_cpu $num_cpu
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python 4_calcDataset.py -file_name $file_name -num_seeds 10 -num_cpu $num_cpu
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python 4_calcDataset.py -file_name $file_name -num_seeds 15 -num_cpu $num_cpu
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python 4_calcDataset.py -file_name $file_name -num_seeds 20 -num_cpu $num_cpu
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python 4_calcDataset.py -file_name $file_name -num_seeds 25 -num_cpu $num_cpu
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $folder/4_calcDataset.py -file_name $file_name -num_seeds 30 -num_cpu $num_cpu
