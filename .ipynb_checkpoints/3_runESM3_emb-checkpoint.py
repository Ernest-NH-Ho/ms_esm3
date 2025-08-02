import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
import torch
import argparse
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import traceback
import sys

with torch.no_grad():
    # model = ESM3(d_model=1536, n_heads=24, v_heads=64, n_layers=48)
    model = torch.load(f"esm3_model/esm3.pth", map_location='cuda').float()

def read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end):
    x = row[x_start:x_end].replace(" ", "")
    y = row[y_start:y_end].replace(" ", "")
    z = row[z_start:z_end].replace(" ", "")
    # print(x, y, z)
    return x, y, z

def read_coord_coarse(row, coords, resi, x_start, x_end, y_start, y_end, z_start, z_end):
    # resi = int(row[23:26].replace(" ", "")) - 1 - trans_pept # - index_reduction
    #if row[12:15].replace(" ", "") == "N":
        # print(f"{read_coord(row)}")
    #    coords[resi-1, 0, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    #elif row[12:15].replace(" ", "") == "CA":
    #if row[12:15].replace(" ", "") == "CA":
        #coords[resi-1, 0, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 0, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 1, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    coords[resi-1, 2, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
        #coords[resi-1, 2, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    #elif row[12:15].replace(" ", "") == "C":
    #    coords[resi-1, 2, :] = read_coord(row, x_start, x_end, y_start, y_end, z_start, z_end)
    return coords

def get_index(title:str):
    idx = {}
    idx["Resi"] = (title.find("RESIDUE"), title.find("RESIDUE")+5)
    idx["Resn"] = (title.find("AA"), title.find("AA")+1)
    idx["SS8"] = (title.find("STRUCTURE"), title.find("STRUCTURE")+1)
    idx["ACC"] = (title.find("ACC"), title.find("ACC")+3)
    idx["Hbond0"] = (title.find("N-H-->O")-2, title.find("N-H-->O")+7)
    idx["Hbond1"] = (title.find("O-->H-N")-2, title.find("O-->H-N")+7)
    idx["Hbond2"] = (title.find("N-H-->O", title.find("N-H-->O")+1)-2, title.find("N-H-->O", title.find("N-H-->O")+1)+7)
    idx["Hbond3"] = (title.find("O-->H-N", title.find("O-->H-N")+1)-2, title.find("O-->H-N", title.find("O-->H-N")+1)+7)
    idx["X-CA"] = (title.find("X-CA")-2, title.find("X-CA")+4)
    idx["Y-CA"] = (title.find("Y-CA")-2, title.find("Y-CA")+4)
    idx["Z-CA"] = (title.find("Z-CA")-2, title.find("Z-CA")+4)
    return idx

def extractNumbers(row, idx, field, sasa_transform=1):
    start, end = idx[field]
    if sasa_transform == None:
        return str(row[start:end].replace(" ", ""))
    else:
        return float(row[start:end].replace(" ", "")) * sasa_transform

def readDSSP(length, cs, entry, retain_residues):
    """ dssp_name is assumed to be f"AF-{entry}-F1-model_v4.dssp"
    length is the length after STP truncation
    return seq, ss8, ACC, coords 
    retain_residues is assumes STP has been cut already """
    # if pdb_name==True:
    #     dssp_filename = f"AF-{entry}-F1-model_v4"
    # else:
    #     dssp_filename = f"{pdb_name}"
    if isinstance(retain_residues, int):
        retain_residues = [retain_residues]
    else:
        retain_residues = retain_residues.split(",")
        retain_residues = [int(i) for i in retain_residues]
    seq_len = 1650
    coords = np.full((1650, 3, 3), np.nan)
    full_seq = ""
    ACC = [None] * 1650
    pLDDT = np.ones((seq_len))
    SS8 = ["_"] * 1650
    struc_token_idx = []
    # print("seq_len =", seq_len)
    length=0
    with open(f"input_dssp/AF-{entry}-F1-model_v4.dssp", "r") as content:
        allrows = content.readlines()
        status = "wait"
        
        for row in allrows:
            # row = row.decode("utf-8")
            if row.count(" # ") > 0:
                idx = get_index(row)
                x_start, x_end = idx["X-CA"]
                y_start, y_end = idx["Y-CA"]
                z_start, z_end = idx["Z-CA"]
                status = "read"
                # print(row)
                
            elif status == "read":
                start, end = idx["Resi"]
                
                if row[start:end].replace(" ", "").isdigit():
                    # print(row)
                    # print("resi in string =", row[start:end])
                    resi = int(row[start:end]) - cs
                    # print("resi in int =", resi)
                    # print(resi)
                    if resi > 0:
                        # SS8
                        start, end = idx["SS8"]
                        if row[start:end] in 'BEGH':
                            SS8[resi-1] = row[start:end]
    
                        # AA
                        resn = extractNumbers(row, idx, "Resn", sasa_transform=None)
                        if resi == 1 and resn == 'M':
                            full_seq += resn
                        else:
                            if resi not in retain_residues:
                                full_seq += "_"
                            else:
                                # If it is essential residue
                                full_seq += resn

                                # # Retain structure token
                                # struc_token_idx.append(resi)
                                
                                # SASA
                                start, end = idx["ACC"]
                                ACC[resi-1] = round(extractNumbers(row, idx, "ACC", sasa_transform=1))
    
                        # Coords
                        coords = read_coord_coarse(row, coords, resi, x_start, x_end, y_start, y_end, z_start, z_end)

                        length += 1
                    
    SS8 = "".join(SS8)
    coords = torch.from_numpy(coords)
    #print("read from dssp", coords)
    protein = ESMProtein()
    protein.sequence, protein.secondary_structure, \
    protein.sasa, protein.plddt, protein.coordinates = full_seq[:length], SS8[:length], ACC[:length], pLDDT[:length], coords[:length, :, :]
    
    return protein #, struc_token_idx

def esm3_generate(model, protein, temp, rs, filename):
    """filename must include folder name """
    # Generate from tensor
    # print("hi")
    protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, \
                                                                        temperature=temp, temperature_annealing=False, \
                                                                        condition_on_coordinates_only = True), \
                               filename)
    # We can show the predicted structure for the generated sequence.
    protein = model.generate(protein, GenerationConfig(track="secondary_structure", num_steps=8, \
                                                                           temperature=temp, temperature_annealing=False, \
                                                                           condition_on_coordinates_only = True), \
                               filename)
    protein = model.decode(protein)
    return protein.sequence


def saveEmbed(protein, model, filename, rs=None):
    if rs==None:
        out_filename = f"esm3_embeddings/wt/{filename}.pt"
    else:
        out_filename = f"esm3_embeddings/rs{rs}/{filename}.pt"
    # print("hi", out_filename)
    
    protein_tensor = model.encode(protein)
    # print(protein_tensor.coordinates)
    # quit()
    # print("hi hi ")
    model(sequence_tokens=protein_tensor.sequence, \
          structure_tokens=protein_tensor.structure, \
          ss8_tokens=protein_tensor.secondary_structure, \
          sasa_tokens=protein_tensor.sasa, \
          structure_coords=protein_tensor.coordinates, \
          filename=out_filename)
    return protein_tensor

def listCritAA(seq, sites):
    output = []
    if isinstance(sites, int):
        sites = [sites]
    else:
        sites = sites.split(",")
        sites = [int(i) for i in sites]
        
    for i in sites:
        output.append(seq[int(i)-1])
    return ",".join(output)
        
def run(row, seed_start, seed_end, model=model):
    try:
        protein = readDSSP(row["Length_final"], row["DSSP_start"], row["Entry"], row["Act_bind_motif_sites"])
        seq_dssp = protein.sequence
        if len(protein.sequence) != len(row["Sequence_final"]):
            with open(f"inconsistentLen.txt", "a+") as out:
                out.write(f"{row['Entry']}\n")
        elif listCritAA(protein.sequence, row["Act_bind_motif_sites"]) != listCritAA(row["Sequence_final"], row["Act_bind_motif_sites"]):
            with open(f"inconsistentSeq.txt", "a+") as out:
                out.write(f"{row['Entry']}\n")
                out.write(f"DSSP\n{protein.sequence}\n")
                out.write(f"UniProtKB\n{row['Sequence_final']}\n")

        protein_tensor = saveEmbed(protein, model, f"AF-{row['Entry']}-F1-model_v4_Emb")
        # print(protein_tensor)
        # quit()
        seq = [seq_dssp]
        for rs in range(seed_start, seed_end):
            torch.manual_seed(rs)
            seq.append(esm3_generate(model, protein_tensor, 0.6, rs, f"esm3_embeddings/rs{rs}/AF-{row['Entry']}-F1-model_v4_Emb_rs{rs}tp06.pt"))
        # print(len(seq))
    except Exception as e:
        with open(f"error.txt", "a+") as out:
            out.write(f"{row}\n")
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n")
            return (None for _ in range(seed_start, seed_end))
    # print(len(seq))
    return (s for s in seq)

def ensure_directories(seed_start, seed_end):
    folder_list = ['wt'] + [f"rs{seed}" for seed in range(seed_start, seed_end)]
    for folder in folder_list:
        if not os.path.exists(f"esm3_embeddings/{folder}"):
            os.mkdir(f"esm3_embeddings/{folder}")
    
parser = argparse.ArgumentParser(description="A script that takes an input value.")
parser.add_argument("-file_name", type=str, required=True, help="Name of UniProt-format dataframe")
parser.add_argument("-num_seeds", type=int, required=True, help="Number of random seed-unique replicate")
args = parser.parse_args()
file_name = args.file_name
num_seeds = args.num_seeds

if __name__ == "__main__":
    with torch.no_grad():
        seed_start = 41
        seed_end = seed_start + num_seeds
        # print(seed_end)
        ensure_directories(seed_start, seed_end)
        DF = pd.read_csv(f"input_csv/{file_name.replace('.csv', '')}.csv", header=0,) # 
        DF["DSSP_start"] = DF["DSSP_start"].astype(str).str.replace(".0", "").astype(int)
        # DF = DF[DF["Entry"]=="F4IK45"]
        with torch.no_grad():
            cols = [f"Sequence_final"] + [f"rs{seed}" for seed in range(seed_start, seed_end)]
            # print(len(cols))
            DF[cols] = DF[["Entry", "DSSP_start", "Length_final", \
                           "Sequence_final", "Act_bind_motif_sites"]].apply(lambda x:pd.Series(run(x, 
                                                                                             seed_start, 
                                                                                             seed_end), 
                                                                                         index=cols),
                                                                      axis=1)
        DF.to_csv(f"input_csv/{file_name.replace('.csv', '')}.csv", index=False)

