# ms_ms3

One-shot structure-based mutation score based on amino acid sequence and PDB backbone coordinates.

## Requirements

### Environment
- Python 3.12
- CUDA 12.4

### Dependencies
```bash
torch==2.5.1
tokenizers==0.20.3
tqdm==4.67.1
scipy==1.14
scikit-learn==1.6
pandas==2.2
numpy==1.26
matplotlib==3.9
einops==0.8.0
cupy==13.4
biopython==1.84
```

## Installation

### Step 1: Set up Conda Environment
Create a new conda environment with Python 3.12 and install the required dependencies listed above.

### Step 2: Install ESM3 Model
1. Obtain the ESM3 model using the official public license from HuggingFace:
   - Visit: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
   - Follow the licensing requirements and download instructions
2. Save the model as `esm_model/esm3.pth` in your project directory

### Step 3: Install DSSP
Download and install DSSP from the official releases:
- https://github.com/cmbi/dssp/releases

### Step 4: Backend Modification
Please refer to `backend_modification.md` for detailed backend setup instructions.

## Usage

### Input File Preparation

1. **PDB Files**: Place input PDB files in the `input_pdb/` directory
2. **CSV Files**: Place input CSV files (in UniProt format) in the `input_csv/` directory

#### Required CSV Fields
The CSV file must contain the following fields:

| Field | Description |
|-------|-------------|
| `Entry` | Sequence ID |
| `Sequence_final` | Valid amino acid sequence string for ESM3 input |
| `Length_final` | Length of the sequence in `Sequence_final` |
| `DSSP_start` | Position of the first PDB residue in `Sequence_final` (0 if sequences are identical, must be ≥ 0) |
| `Act_bind_motif_sites` | Critical residues in `Sequence_final` (first residue has index 1) |

**Important Notes:**
- Make sure the PDB file name has is AF-<Entry>-F1-model_v4.pdb 
- Make sure each sequence ID is unique in the csv
- Do not include sequences over 1,650 amino acids for ESM3 input
- Ensure CSV sequences are identical to PDB sequences after accounting for the shifts in "DSSP_start"

### Running the Analysis

```bash
./lazy_run.sh --file_name demo --num_seeds 30 --num_cpu 4
./lazy_run.sh --help
```

#### Script Execution Order
When running `./lazy_run.sh`, the following scripts execute in chronological order:

1. **`1_dssp.sh`** - Generate DSSP from PDB files
2. **`2_processData.py`** - *Optional*: Process sequences with signal/transit peptides or non-identical PDB/CSV sequences
3. **`3_runESM3_emb.py`** - Generate de novo sequences using ESM3
4. **`4_calcDataset.py`** - Output data analysis summary and tensors


## Citation
To be confirmed (TBC)

## Support
Ngai Hei Ernest Ho: hoernesta@gmail.com
