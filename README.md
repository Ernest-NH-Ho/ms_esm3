# ms_ms3

One-shot structure-based mutation score based on amino acid sequence and PDB backbone coordinate.

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

1. Place the input pdb files in the "input_pdb/"
2. Place the input csv files in UniProt format in "input_csv"

## Citation
tbc
